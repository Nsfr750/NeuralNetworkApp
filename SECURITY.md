# Security Policy

## Supported Versions

We follow semantic versioning and provide security updates for the following versions:

| Version | Supported          | Security Updates |
| ------- | ------------------ | ---------------- |
| main    | :white_check_mark: | Yes              |
| latest  | :white_check_mark: | Yes              |
| 1.x.x   | :white_check_mark: | Critical only    |
| < 1.0   | :x:                | No               |

Security patches are applied to the latest stable version and backported to supported minor versions when feasible.

## Security Scope

This application handles several types of sensitive data and operations:

### Data Security

- **User Datasets**: Training data, validation data, and test datasets
- **Model Files**: Trained neural network models and checkpoints
- **Configuration Files**: Network architectures, hyperparameters, and training settings
- **Log Files**: Application logs that may contain sensitive information

### Application Security

- **GUI Interface**: User input handling and display
- **File Operations**: Reading/writing models and datasets
- **Network Operations**: Update checking and potential future cloud features
- **System Integration**: File system access and resource management

## Known Security Considerations

### Data Privacy

- **Local Storage**: All data is stored locally by default
- **No Data Collection**: The application does not collect or transmit user data
- **Temporary Files**: Dataset processing may create temporary files in the `data/temp/` directory
- **Model Serialization**: Models are saved using PyTorch's native serialization

### Dependency Security

- **PyTorch**: Core ML framework with regular security updates
- **PySide6**: GUI framework for user interface
- **Wand**: Image processing library (used instead of PIL/Pillow)
- **NumPy/Pandas**: Data processing libraries
- **Third-party Packages**: All dependencies are listed in `requirements.txt`

### File System Access

- **Read Access**: Can read datasets, models, and configuration files
- **Write Access**: Creates models, logs, checkpoints, and temporary files
- **Directory Structure**: Follows organized directory structure for different file types

## Security Best Practices

### For Users

1. **Data Sources**: Only use datasets from trusted sources
2. **Model Sharing**: Be cautious when sharing trained models that may contain sensitive training data
3. **File Permissions**: Ensure appropriate file permissions for your datasets and models
4. **Regular Updates**: Keep the application and dependencies updated
5. **Backup Strategy**: Maintain backups of important models and datasets

### For Development

1. **Input Validation**: All user inputs are validated before processing
2. **Error Handling**: Comprehensive error handling prevents information leakage
3. **Logging**: Sensitive information is not logged; logs are rotated and managed
4. **Dependency Management**: Regular security updates for all dependencies
5. **Code Review**: All code changes undergo security review

## Vulnerability Reporting

### How to Report

If you discover a security vulnerability, please report it responsibly:

- **Primary Contact**: [Nsfr750](mailto:nsfr750@yandex.com)
- **GitHub Security**: <https://github.com/Nsfr750/NeuralNetworkApp/security/advisories/new>
- **Discord**: <https://discord.gg/ryqNeuRYjD>

### What to Include

Please include the following information in your report:

- Vulnerability type and description
- Steps to reproduce the issue
- Expected vs. actual behavior
- Potential impact
- Environment details (OS, Python version, application version)
- Proof of concept (if applicable)

### Response Timeline

- **Acknowledgment**: Within 48 hours of receipt
- **Initial Assessment**: Within 3-5 business days
- **Patch Development**: Timeline depends on severity
- **Public Disclosure**: After patch is available and users have had time to update

### Severity Levels

| Severity | Description | Response Time |
| -------- | ----------- | ------------- |
| Critical | Remote code execution, data exposure | 24-48 hours |
| High | Privilege escalation, significant data leak | 3-5 days |
| Medium | Limited data exposure, denial of service | 1-2 weeks |
| Low | Minor information disclosure, UI issues | Next release |

## Security Features

### Built-in Protections

- **Input Sanitization**: All file paths and user inputs are sanitized
- **Safe File Operations**: Secure file handling with proper error checking
- **Memory Management**: Proper cleanup of resources and temporary files
- **Exception Handling**: Secure error handling that doesn't expose sensitive information

### Data Protection

- **Local Processing**: All neural network training happens locally
- **No Telemetry**: Application does not send usage data or analytics
- **Configurable Logging**: Log levels can be adjusted to reduce sensitive information exposure
- **Temporary File Cleanup**: Automatic cleanup of temporary processing files

## Dependencies Security

### Regular Updates

- Dependencies are regularly monitored for security updates
- Critical security patches are applied promptly
- Version ranges in requirements.txt balance stability and security

### Vulnerability Scanning

- Regular security scanning of dependencies
- Integration with security advisory databases
- Automated checks for known vulnerabilities

## Incident Response

### Types of Incidents

- **Security Vulnerabilities**: Exploitable weaknesses in the application
- **Data Breaches**: Unauthorized access to sensitive data
- **Denial of Service**: Attacks affecting application availability
- **Malware Detection**: Compromised builds or dependencies

### Response Process

1. **Identification**: Detect and confirm the security incident
2. **Containment**: Limit the impact and prevent further damage
3. **Analysis**: Investigate root cause and scope
4. **Remediation**: Develop and deploy fixes
5. **Communication**: Notify affected users and stakeholders
6. **Prevention**: Implement measures to prevent recurrence

## Legal and Compliance

### Privacy Policy

- No personal data is collected or transmitted
- All processing happens locally on user's machine
- Users maintain full control over their data and models

### License Compliance

- Application is licensed under GPLv3
- All dependencies comply with their respective licenses
- License information is provided in the LICENSE file

### Export Controls

- Neural network models may be subject to export regulations
- Users are responsible for compliance with applicable laws
- No built-in restrictions on model capabilities

## Contact Information

For security-related inquiries:

- **Email**: [Nsfr750](mailto:nsfr750@yandex.com)
- **GitHub**: <https://github.com/Nsfr750/NeuralNetworkApp/security/advisories>
- **Discord**: <https://discord.gg/ryqNeuRYjD>
- **Documentation**: [Security Guide](docs/guides/security.md)

## Acknowledgments

We thank the security research community for their efforts in making this application more secure. All responsible vulnerability disclosures are appreciated and will be acknowledged appropriately.
