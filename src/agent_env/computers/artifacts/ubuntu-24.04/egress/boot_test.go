package main

import (
	"os"
	"os/user"
	"path/filepath"
	"strings"
	"testing"
)

func TestBootMintsTheCAOnceItsKeyPrivateAndTrustsItInEveryBundleOnce(t *testing.T) {
	dir := t.TempDir()
	me, _ := user.Current()
	bundle := filepath.Join(dir, "ca-bundle.pem")
	os.WriteFile(bundle, []byte("-----BEGIN CERTIFICATE-----\nroot\n-----END CERTIFICATE-----"), 0o644)
	certifi := filepath.Join(dir, "opt", "venv", "lib", "certifi", "cacert.pem")
	os.MkdirAll(filepath.Dir(certifi), 0o755)
	os.WriteFile(certifi, []byte(""), 0o644)
	p := bootPaths{
		Folder: filepath.Join(dir, "egress"), PublicCA: filepath.Join(dir, "egress-ca.pem"),
		Anchor:       filepath.Join(dir, "anchors", "agentenv-egress.crt"),
		Bundles:      []string{bundle, filepath.Join(dir, "missing.pem")},
		CertifiRoots: []string{filepath.Join(dir, "opt")},
		EgressUser:   me.Username,
	}
	report, err := boot(p)
	if err != nil {
		t.Fatal(err, report)
	}
	if report["minted"] != true {
		t.Fatalf("%v", report)
	}
	info, _ := os.Stat(filepath.Join(p.Folder, "ca.key"))
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("the CA key is %v", info.Mode().Perm())
	}
	if info, _ := os.Stat(p.Folder); info.Mode().Perm() != 0o700 {
		t.Fatalf("the folder is %v", info.Mode().Perm())
	}
	ca, _ := os.ReadFile(filepath.Join(p.Folder, "ca.pem"))
	public, _ := os.ReadFile(p.PublicCA)
	if string(ca) != string(public) {
		t.Fatal("the public CA isn't the CA")
	}
	if _, err := boot(p); err != nil {
		t.Fatal(err)
	}
	again, _ := os.ReadFile(filepath.Join(p.Folder, "ca.pem"))
	if string(again) != string(ca) {
		t.Fatal("a second boot minted another CA")
	}
	for _, path := range []string{bundle, certifi} {
		text, _ := os.ReadFile(path)
		if n := strings.Count(string(text), strings.TrimSpace(string(ca))); n != 1 {
			t.Fatalf("%s has the CA %d times", path, n)
		}
	}
	leaves, err := LoadLeaves(p.Folder)
	if err != nil || leaves.ca.cert.NotBefore != CAFrom || leaves.ca.cert.NotAfter != CAUntil {
		t.Fatalf("%v", err)
	}
}

func TestAnAppsStoreThatCantTakeTheCAIsNamedAndTheBootGoesOn(t *testing.T) {
	dir := t.TempDir()
	me, _ := user.Current()
	broken := filepath.Join(dir, "opt", "venv", "lib", "certifi")
	os.MkdirAll(filepath.Join(broken, "elsewhere"), 0o755)
	os.Symlink(filepath.Join(broken, "elsewhere"), filepath.Join(broken, "cacert.pem")) // a folder, not a bundle
	jdk := filepath.Join(dir, "jdk", "lib", "security")
	os.MkdirAll(jdk, 0o755)
	os.WriteFile(filepath.Join(jdk, "cacerts"), []byte("not a keystore"), 0o644)
	t.Setenv("PATH", dir) // and no keytool to add the CA with
	p := bootPaths{
		Folder: filepath.Join(dir, "egress"), PublicCA: filepath.Join(dir, "egress-ca.pem"),
		Anchor:       filepath.Join(dir, "anchors", "agentenv-egress.crt"),
		CertifiRoots: []string{filepath.Join(dir, "opt")}, JavaRoots: []string{filepath.Join(dir, "jdk")},
		EgressUser: me.Username,
	}
	report, err := boot(p)
	if err != nil {
		t.Fatalf("the boot failed: %v", err)
	}
	warnings, _ := report["warnings"].([]string)
	if len(warnings) != 2 || !strings.Contains(strings.Join(warnings, " "), "keytool") {
		t.Fatalf("%v", report)
	}
}
