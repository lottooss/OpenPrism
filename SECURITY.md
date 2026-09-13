# Security Policy

## Supported code

Security fixes target the `main` branch until versioned releases exist.

## Reporting

Do not open a public issue for a vulnerability involving credentials, arbitrary native plugin/engine loading, shared-memory validation, input safety bypass, or destructive behavior. Report it privately to the repository owner through GitHub Security Advisories once the remote repository is configured.

## Trust boundaries

- Native plugins and TensorRT engines are executable code and must be hash-verified and loaded only from trusted locations.
- External/shared-memory messages are untrusted until bounds and schema verification succeeds.
- Actuation must always pass focus, freshness, calibration, bounds, uncertainty, and emergency-stop gates.
- Secrets, private recordings, datasets, and credentials must not enter Git, CI logs, issues, or pull requests.

This project does not accept features for process injection, game-memory inspection, anti-cheat bypass, or concealment of automated behavior.
