# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues, discussions or
pull requests. Report them privately, in either of two ways:

- through GitHub: open this repository's **Security** tab and choose **Report a vulnerability**;
- to Scale's vulnerability disclosure team, as described under "Responsible Disclosure" at
  https://scale.com/legal/security.

Include:

- a description of the issue and the component involved (module, task step, sandbox provider);
- the affected version or commit;
- steps or a proof of concept to reproduce it;
- the impact you believe it has.

We will acknowledge your report and work with you on a fix and a coordinated disclosure. Please
give us reasonable time to address the issue before disclosing it publicly.

## Supported versions

Security fixes land on `main` and ship in the next release. Only the latest release is supported;
please reproduce against it before reporting.

## Scope

agent-env orchestrates agents and environments inside sandboxes run by third-party or
self-hosted backends. Weaknesses in how agent-env drives those sandboxes, handles credentials,
verifies TLS, or isolates one run from another are in scope. Vulnerabilities in the sandbox
platforms themselves should be reported to their vendors.
