# Security Policy

## Supported Versions

Edge Voice Assistant (EVA) is currently in **Alpha** status.

We provide security updates for the current minor release line. Please ensure you are testing against the latest version on the `main` branch or the latest tagged release.

| Version | Supported |
|---------|-----------|
| 0.7.x-alpha | :white_check_mark: |
| < 0.7.x | :x: |

## Scope

EVA is designed as a local-first, offline voice assistant. The policy broadly covers security vulnerabilities affecting EVA, including:
- Application and runtime privilege escalation or code execution
- Local API and frontend boundaries
- Filesystem and data handling (including semantic memory storage)
- Model and runtime integration
- Audio processing integrity
- Configuration and settings management
- Dependencies affecting the runtime
- Plugin and tool execution surfaces (when those features actually ship)

**Note:** Runtime tool execution, online web search, external API provider integrations, and multimodal capabilities are **NOT** currently implemented. They remain future additions. Please do not report hypothetical vulnerabilities for these unreleased capabilities until they exist in the repository.

## Reporting a Vulnerability

We take the security of this project seriously. Please **DO NOT** disclose vulnerabilities publicly until we have had a chance to coordinate a fix.

If **GitHub Private Vulnerability Reporting** is enabled for this repository (under the **Security > Advisories** tab), please use it to privately notify the maintainers.

If Private Vulnerability Reporting is unavailable, please contact the repository owner (`FahadiF`) privately via direct message or another private channel before opening a public issue.

### What to include in your report
To help us resolve the issue quickly, please include the following information:
- A detailed description of the vulnerability and its potential impact.
- The version of EVA and your operating system.
- Step-by-step instructions to reproduce the issue.
- Expected versus actual behavior.
- Any relevant logs, crash dumps, or error messages (please remove any personal or sensitive information before submitting).

### What to expect
- You should receive an acknowledgement of your report within 72 hours.
- We will investigate the issue and communicate our assessment.
- Since this is an open-source project maintained voluntarily, we cannot guarantee an immediate SLA for a fix, but we will prioritize critical vulnerabilities affecting the offline integrity of the assistant.
- We ask for your patience and cooperation in keeping the details private until a patch is released.
