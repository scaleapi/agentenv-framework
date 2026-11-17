package main

import (
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// bootPaths are where boot reads and writes; tests point them into a folder of their own.
type bootPaths struct {
	Folder, PublicCA, Anchor string
	// Bundles are the PEM bundles the CA is appended to when they exist: agent-env's trust files, which
	// SSL_CERT_FILE, REQUESTS_CA_BUNDLE and NODE_EXTRA_CA_CERTS name.
	Bundles []string
	// UpdateTrust rebuilds the system store from its anchors, and its hooks Java's.
	UpdateTrust []string
	// CertifiRoots and JavaRoots are searched for certifi's cacert.pem and JDKs' lib/security/cacerts.
	CertifiRoots, JavaRoots []string
	EgressUser, SessionUser string
	// NSS adds the CA to the session user's NSS database, which Chromium and Firefox read; nil leaves it.
	NSS func(paths bootPaths) error
}

func defaultBootPaths() bootPaths {
	return bootPaths{
		Folder: Folder, PublicCA: PublicCA, Anchor: "/usr/local/share/ca-certificates/agentenv-egress.crt",
		Bundles:      []string{"/etc/agentenv/ca.pem", "/etc/agentenv/ca-bundle.pem"},
		UpdateTrust:  []string{"update-ca-certificates"},
		CertifiRoots: []string{"/usr/lib/python3", "/usr/local/lib", "/opt"},
		JavaRoots:    []string{"/usr/lib/jvm", "/opt"},
		EgressUser:   EgressUser, SessionUser: SessionUser,
		NSS: sessionNSS,
	}
}

// boot, as root before the session starts: the CA, minted once per computer, its key the egress user's alone, and
// trusted in the system store, agent-env's trust files, every certifi bundle, every JDK's store and the session
// user's NSS database. A step that fails fails the boot, naming it, but for a certifi bundle or a JDK's store, which
// an app brings: those are named among the report's warnings.
func boot(p bootPaths) (map[string]any, error) {
	start := time.Now()
	report := map[string]any{}
	egress, err := user.Lookup(p.EgressUser)
	if err != nil {
		return report, err
	}
	uid, _ := strconv.Atoi(egress.Uid)
	gid, _ := strconv.Atoi(egress.Gid)
	if err := os.MkdirAll(p.Folder, 0o700); err != nil {
		return report, err
	}
	caPath, keyPath := filepath.Join(p.Folder, "ca.pem"), filepath.Join(p.Folder, "ca.key")
	if _, err := os.Stat(caPath); errors.Is(err, fs.ErrNotExist) {
		if err := writeCA(caPath, keyPath); err != nil {
			return report, fmt.Errorf("minting the CA: %w", err)
		}
		report["minted"] = true
	}
	for _, path := range []string{p.Folder, caPath, keyPath} {
		if err := os.Chown(path, uid, gid); err != nil {
			return report, err
		}
	}
	if err := os.Chmod(p.Folder, 0o700); err != nil {
		return report, err
	}
	caPEM, err := os.ReadFile(caPath)
	if err != nil {
		return report, err
	}
	for _, path := range []string{p.PublicCA, p.Anchor} {
		if err := writeRootFile(path, caPEM); err != nil {
			return report, err
		}
	}
	if len(p.UpdateTrust) > 0 {
		if out, err := exec.Command(p.UpdateTrust[0], p.UpdateTrust[1:]...).CombinedOutput(); err != nil {
			return report, fmt.Errorf("%s: %w: %s", strings.Join(p.UpdateTrust, " "), err, truncate(string(out), ErrorLimit))
		}
	}
	appended := map[string]bool{}
	for _, path := range p.Bundles {
		added, err := appendOnce(path, caPEM)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return report, fmt.Errorf("trusting the CA in %s: %w", path, err)
		}
		appended[path] = added
	}
	// An app's own stores, certifi's and JDKs': one that can't take the CA is named, and the computer boots, its
	// app then refusing intercepted sites as it would on a store of its own.
	var warnings []string
	for _, path := range certifiBundles(p.CertifiRoots) {
		added, err := appendOnce(path, caPEM)
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("trusting the CA in %s: %v", path, err))
			continue
		}
		appended[path] = added
	}
	report["bundles"] = appended
	java := map[string]string{}
	for _, store := range javaStores(p.JavaRoots) {
		if err := trustJava(store, p.PublicCA); err != nil {
			warnings = append(warnings, fmt.Sprintf("trusting the CA in %s: %v", store, err))
			continue
		}
		java[store] = "trusted"
	}
	report["java"] = java
	if len(warnings) > 0 {
		report["warnings"] = warnings
	}
	if p.NSS != nil {
		if err := p.NSS(p); err != nil {
			return report, fmt.Errorf("the session's NSS database: %w", err)
		}
		report["nss"] = "trusted"
	}
	report["seconds"] = time.Since(start).Seconds()
	return report, nil
}

func writeCA(caPath, keyPath string) error {
	ca, err := mintCA(CAName)
	if err != nil {
		return err
	}
	key, err := x509.MarshalPKCS8PrivateKey(ca.key)
	if err != nil {
		return err
	}
	if err := writeExclusive(keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key}), 0o600); err != nil {
		return err
	}
	return writeExclusive(caPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: ca.cert.Raw}), 0o644)
}

func writeExclusive(path string, data []byte, mode os.FileMode) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// writeRootFile writes path whole, 0644, replacing it in one rename.
func writeRootFile(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	part := path + ".part"
	if err := os.WriteFile(part, data, 0o644); err != nil {
		return err
	}
	if err := os.Chmod(part, 0o644); err != nil {
		return err
	}
	return os.Rename(part, path)
}

// appendOnce appends pem to the bundle at path (through a symlink) unless it's there; whether it did.
func appendOnce(path string, pemData []byte) (bool, error) {
	real, err := filepath.EvalSymlinks(path)
	if err != nil {
		return false, err
	}
	text, err := os.ReadFile(real)
	if err != nil {
		return false, err
	}
	block := strings.TrimSpace(string(pemData))
	if strings.Contains(string(text), block) {
		return false, nil
	}
	f, err := os.OpenFile(real, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return false, err
	}
	defer f.Close()
	sep := ""
	if len(text) > 0 && !strings.HasSuffix(string(text), "\n") {
		sep = "\n"
	}
	_, err = f.WriteString(sep + block + "\n")
	return err == nil, err
}

func certifiBundles(roots []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, root := range roots {
		filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() || d.Name() != "cacert.pem" || filepath.Base(filepath.Dir(path)) != "certifi" {
				return nil
			}
			real, err := filepath.EvalSymlinks(path)
			if err == nil && !seen[real] {
				seen[real] = true
				out = append(out, path)
			}
			return nil
		})
	}
	sort.Strings(out)
	return out
}

// javaStores are the JDKs' stores, lib/security/cacerts, under roots; the system store's hook builds Debian's.
func javaStores(roots []string) []string {
	var out []string
	for _, root := range roots {
		filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() || d.Name() != "cacerts" {
				return nil
			}
			if filepath.Base(filepath.Dir(path)) == "security" && filepath.Base(filepath.Dir(filepath.Dir(path))) == "lib" {
				if info, err := os.Lstat(path); err == nil && info.Mode().IsRegular() {
					out = append(out, path)
				}
			}
			return nil
		})
	}
	sort.Strings(out)
	return out
}

// trustJava adds the CA to a JDK's store with that JDK's own keytool, else any on the PATH.
func trustJava(store, ca string) error {
	keytool := filepath.Join(filepath.Dir(filepath.Dir(filepath.Dir(store))), "bin", "keytool")
	if _, err := os.Stat(keytool); err != nil {
		if keytool, err = exec.LookPath("keytool"); err != nil {
			return errors.New("there's no keytool to add the CA with")
		}
	}
	opened := []string{"-keystore", store, "-storepass", "changeit"}
	exec.Command(keytool, append([]string{"-delete", "-alias", "agentenv-egress"}, opened...)...).Run()
	out, err := exec.Command(keytool, append([]string{"-importcert", "-noprompt", "-trustcacerts", "-alias",
		"agentenv-egress", "-file", ca}, opened...)...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("keytool: %w: %s", err, truncate(string(out), ErrorLimit))
	}
	return nil
}

// sessionNSS adds the CA to the session user's NSS database, as that user, making the database if there's none.
func sessionNSS(p bootPaths) error {
	session, err := user.Lookup(p.SessionUser)
	if err != nil {
		return err
	}
	db := filepath.Join(session.HomeDir, ".pki", "nssdb")
	as := []string{"setpriv", "--reuid=" + p.SessionUser, "--regid=" + p.SessionUser, "--init-groups", "env",
		"HOME=" + session.HomeDir}
	steps := [][]string{{"mkdir", "-p", db}}
	if _, err := os.Stat(filepath.Join(db, "cert9.db")); err != nil {
		steps = append(steps, []string{"certutil", "-N", "-d", "sql:" + db, "--empty-password"})
	}
	steps = append(steps, []string{"certutil", "-A", "-d", "sql:" + db, "-n", NSSNickname, "-t", "C,,", "-i",
		p.PublicCA})
	for _, step := range steps {
		argv := append(append([]string{}, as...), step...)
		if out, err := exec.Command(argv[0], argv[1:]...).CombinedOutput(); err != nil {
			return fmt.Errorf("%s: %w: %s", strings.Join(step[:2], " "), err, truncate(string(out), ErrorLimit))
		}
	}
	return nil
}
