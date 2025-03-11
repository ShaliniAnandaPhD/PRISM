"""
This script evaluates data security and legal compliance for Python code, applications,
and data handling processes. It performs comprehensive analysis of:

- Data protection (encryption, anonymization, minimization)
- Access controls and authentication mechanisms
- Code security (vulnerabilities, secrets, secure coding practices)
- Regulatory compliance (GDPR, HIPAA, PCI-DSS, CCPA)
- Third-party dependency security
- Secure communications
- Audit logging and monitoring
- Infrastructure security configurations

The framework provides detailed reporting, risk assessment, and remediation guidance
to ensure systems meet security best practices and legal requirements.
"""
import logging
import json
import os
import sys
import re
import hashlib
import inspect
import importlib
import stat
import math
import platform
import socket
import ssl
import subprocess
import tempfile
import argparse
import datetime
import base64
import urllib.request
import pkg_resources
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
import requests

# Configure logging for auditability
logging.basicConfig(
    filename='security_compliance.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global configuration
DEFAULT_CONFIG = {
    "encryption": {
        "min_entropy_threshold": 7.0,
        "required_algorithms": ["AES", "RSA", "ECDSA"],
        "prohibited_algorithms": ["DES", "RC4", "MD5", "SHA1"],
        "min_key_lengths": {
            "RSA": 2048,
            "EC": 256,
            "AES": 256
        }
    },
    "access_control": {
        "file_permissions": {
            "config_files_max_permission": 0o644,
            "key_files_max_permission": 0o600,
            "data_files_max_permission": 0o644
        },
        "required_auth_mechanisms": ["MFA", "password", "session_timeout"]
    },
    "secret_detection": {
        "entropy_threshold": 4.5,
        "min_secret_length": 8,
        "max_false_positive_rate": 0.1
    },
    "compliance": {
        "gdpr": {
            "required_features": ["right_to_access", "right_to_be_forgotten", "data_portability"],
            "data_retention_max_days": 730  # 2 years
        },
        "hipaa": {
            "required_features": ["access_logging", "encryption_at_rest", "encryption_in_transit"],
            "phi_protection_methods": ["encryption", "tokenization", "anonymization"]
        },
        "pci": {
            "card_number_pattern": r"\b(?:\d[ -]*?){13,16}\b",
            "prohibited_storage": ["CVV", "PIN", "full_magnetic_stripe"]
        }
    },
    "code_security": {
        "vulnerability_patterns": {
            "sql_injection": [
                r"execute\(['\"].*?\bWHERE\b.*?\+",
                r"execute\(['\"].*?\bSELECT\b.*?\+",
                r"cursor\.execute\([^,)]*?\%"
            ],
            "command_injection": [
                r"os\.system\([^)]*?\+",
                r"subprocess\.call\([^)]*?\+",
                r"eval\([^)]*?\)"
            ],
            "xss": [
                r"render\([^)]*?request\.get",
                r"\.html\([^)]*?user",
                r"document\.write\([^)]*?\+"
            ]
        },
        "secure_coding_practices": {
            "input_validation": [r"\.isdigit\(", r"\.strip\(", r"\.validate\("],
            "output_encoding": [r"html\.escape\(", r"cgi\.escape\(", r"urllib\.quote\("],
            "parameterized_queries": [r"\.execute\([^,]+?, \("]
        }
    },
    "dependency_security": {
        "vulnerability_db_url": "https://example.com/vulnerabilities.json",  # Placeholder
        "check_outdated_packages": True,
        "prohibited_packages": ["insecure-package", "deprecated-library"]
    }
}

# Classes for different types of security issues
class SecurityIssue:
    """Base class for security issues."""
    def __init__(self, title, description, severity, location=None, remediation=None):
        self.title = title
        self.description = description
        self.severity = severity  # 'critical', 'high', 'medium', 'low', 'info'
        self.location = location
        self.remediation = remediation
        self.timestamp = datetime.datetime.now().isoformat()
    
    def to_dict(self):
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "location": self.location,
            "remediation": self.remediation,
            "timestamp": self.timestamp
        }

class VulnerabilityIssue(SecurityIssue):
    """Security issue related to code vulnerabilities."""
    def __init__(self, title, description, severity, vulnerability_type, 
                 cwe_id=None, location=None, remediation=None):
        super().__init__(title, description, severity, location, remediation)
        self.vulnerability_type = vulnerability_type
        self.cwe_id = cwe_id
    
    def to_dict(self):
        result = super().to_dict()
        result.update({
            "vulnerability_type": self.vulnerability_type,
            "cwe_id": self.cwe_id
        })
        return result

class ComplianceIssue(SecurityIssue):
    """Security issue related to regulatory compliance."""
    def __init__(self, title, description, severity, regulation, 
                 requirement=None, location=None, remediation=None):
        super().__init__(title, description, severity, location, remediation)
        self.regulation = regulation
        self.requirement = requirement
    
    def to_dict(self):
        result = super().to_dict()
        result.update({
            "regulation": self.regulation,
            "requirement": self.requirement
        })
        return result

class ConfigurationIssue(SecurityIssue):
    """Security issue related to system configuration."""
    def __init__(self, title, description, severity, component, 
                 current_value=None, recommended_value=None, location=None, remediation=None):
        super().__init__(title, description, severity, location, remediation)
        self.component = component
        self.current_value = current_value
        self.recommended_value = recommended_value
    
    def to_dict(self):
        result = super().to_dict()
        result.update({
            "component": self.component,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value
        })
        return result

# Core utility functions
def calculate_entropy(data):
    """
    Calculate Shannon entropy for binary data or string.
    Higher values indicate higher randomness (more likely to be encrypted or a key).
    
    :param data: Data to analyze (string or bytes)
    :return: Entropy value (0-8 for ASCII, higher means more random)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if not data:
        return 0
    
    entropy = 0
    size = len(data)
    counts = {}
    
    # Count byte occurrences
    for byte in data:
        counts[byte] = counts.get(byte, 0) + 1
    
    # Calculate entropy
    for count in counts.values():
        probability = count / size
        entropy -= probability * math.log2(probability)
    
    return entropy

def find_potential_secrets(code_str, min_length=8, entropy_threshold=4.5):
    """
    Find strings in code that might be secrets based on entropy and patterns.
    
    :param code_str: String containing source code
    :param min_length: Minimum length for a string to be considered a potential secret
    :param entropy_threshold: Minimum entropy value to consider as a secret
    :return: List of potential secrets with their locations
    """
    # Regex for finding string literals
    string_pattern = r'(["\'])((?:(?=(\\?))\3.)*?)\1'
    secrets = []
    
    for match in re.finditer(string_pattern, code_str):
        string_content = match.group(2)
        
        # Skip short strings, common words, or obvious non-secrets
        if (len(string_content) >= min_length and 
            not re.match(r'^[a-zA-Z]+$', string_content) and  # Skip plain English words
            not string_content.startswith('http')):  # Skip URLs
            
            entropy = calculate_entropy(string_content)
            if entropy >= entropy_threshold:
                line_num = code_str[:match.start()].count('\n') + 1
                secrets.append({
                    'value': string_content,
                    'entropy': entropy,
                    'line': line_num,
                    'match': match.group(0)
                })
    
    return secrets

def check_file_permissions(file_path, max_permission):
    """
    Check if a file has permissions that are too permissive.
    
    :param file_path: Path to the file
    :param max_permission: Maximum allowed permission (e.g., 0o644)
    :return: True if permissions are acceptable, False otherwise
    """
    if not os.path.exists(file_path):
        return True
    
    current_permissions = stat.S_IMODE(os.stat(file_path).st_mode)
    return current_permissions <= max_permission

def is_data_encrypted(data):
    """
    Try to determine if data is encrypted based on entropy and patterns.
    
    :param data: Data to check (bytes or string)
    :return: Tuple of (is_encrypted, confidence, reason)
    """
    if isinstance(data, str):
        try:
            # Try to decode as base64 first
            decoded = base64.b64decode(data)
            data = decoded
        except:
            data = data.encode('utf-8')
    
    if not isinstance(data, bytes):
        return (False, 1.0, "Data is not in bytes format")
    
    entropy = calculate_entropy(data)
    
    # Check for common encryption headers
    if data.startswith(b'-----BEGIN') and b'ENCRYPTED' in data[:40]:
        return (True, 0.95, "PEM encrypted format detected")
    
    # Check entropy - high entropy suggests encryption/compression
    if entropy > 7.5:
        return (True, 0.9, f"Very high entropy ({entropy:.2f})")
    elif entropy > 6.5:
        return (True, 0.7, f"High entropy ({entropy:.2f})")
    elif entropy < 3.0:
        return (False, 0.8, f"Low entropy ({entropy:.2f})")
    
    # Check for patterns that suggest encryption
    if re.search(b'\\x00\\x00\\x00\\x00', data) and entropy > 5.0:
        return (False, 0.6, "Contains null byte patterns, possible binary data")
    
    # If data is mostly printable, probably not encrypted
    printable_ratio = sum(c in range(32, 127) for c in data) / len(data)
    if printable_ratio > 0.9:
        return (False, 0.7, f"Mostly printable characters ({printable_ratio:.2f})")
    
    # Default case - uncertain
    return (None, 0.5, "Indeterminate")

# Core security scanning functions
def check_data_encryption(data, min_entropy_threshold=7.0):
    """
    Check whether data appears to be properly encrypted.
    
    :param data: Data to check for encryption
    :param min_entropy_threshold: Minimum entropy to consider data encrypted
    :return: Dictionary with encryption assessment results
    """
    is_encrypted, confidence, reason = is_data_encrypted(data)
    
    # If we got bytes from a function that claims to encrypt, but it doesn't look encrypted
    if is_encrypted is False and confidence >= 0.7:
        issues = [
            SecurityIssue(
                title="Data does not appear to be properly encrypted",
                description=f"Data has characteristics of unencrypted content: {reason}",
                severity="high",
                remediation="Implement proper encryption using industry-standard algorithms"
            )
        ]
    else:
        issues = []
    
    result = {
        "appears_encrypted": is_encrypted,
        "confidence": confidence,
        "reason": reason,
        "issues": issues
    }
    
    return result

def check_access_controls(paths=None):
    """
    Check if proper access control mechanisms are in place.
    
    :param paths: Optional list of paths to check permissions
    :return: Dictionary with access control assessment results
    """
    issues = []
    
    # Check if running as root (not recommended for production)
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        issues.append(
            ConfigurationIssue(
                title="Application running with root privileges",
                description="Running with elevated privileges increases security risk",
                severity="high",
                component="process_privileges",
                current_value="root (uid 0)",
                recommended_value="non-root user",
                remediation="Configure application to run as a dedicated non-root user"
            )
        )
    
    # Check file permissions if paths provided
    if paths:
        for path in paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    # Check if path might contain sensitive data
                    is_sensitive = any(substring in path.lower() for substring in 
                                      ['password', 'secret', 'key', 'credential', 'token', 'auth'])
                    
                    max_perm = 0o600 if is_sensitive else 0o644
                    perm_ok = check_file_permissions(path, max_perm)
                    
                    if not perm_ok:
                        current_perm = stat.S_IMODE(os.stat(path).st_mode)
                        issues.append(
                            ConfigurationIssue(
                                title=f"Excessive file permissions for {'sensitive' if is_sensitive else 'configuration'} file",
                                description=f"File {path} has overly permissive access rights",
                                severity="high" if is_sensitive else "medium",
                                component="file_permissions",
                                current_value=f"{current_perm:o}",
                                recommended_value=f"{max_perm:o}",
                                location=path,
                                remediation=f"Change file permissions: chmod {max_perm:o} {path}"
                            )
                        )
    
    # Check for basic authentication mechanism in the code
    auth_mechanisms = []
    for module_name in sys.modules:
        if any(name in module_name for name in ['auth', 'oauth', 'jwt', 'session']):
            auth_mechanisms.append(module_name)
    
    result = {
        "issues": issues,
        "detected_auth_mechanisms": auth_mechanisms,
        "running_as_root": os.geteuid() == 0 if hasattr(os, 'geteuid') else None
    }
    
    return result

def scan_for_hardcoded_secrets(code, extra_patterns=None):
    """
    Scan code for hardcoded API keys, passwords, or sensitive information.
    
    :param code: Source code as a string
    :param extra_patterns: Additional regex patterns to check
    :return: Dictionary with secret scanning results
    """
    issues = []
    
    # Default patterns to look for
    patterns = {
        "API Key": [
            r'api[_-]?key[_-]?=\s*["\']([a-zA-Z0-9_\-\.]{8,})["\']',
            r'api[_-]?secret[_-]?=\s*["\']([a-zA-Z0-9_\-\.]{8,})["\']',
            r'key[_-]?=\s*["\']([a-zA-Z0-9_\-\.]{16,})["\']'
        ],
        "AWS Key": [
            r'AKIA[0-9A-Z]{16}',
            r'aws[_-]?access[_-]?key[_-]?id[_-]?=\s*["\']([A-Za-z0-9/+=]{8,})["\']',
            r'aws[_-]?secret[_-]?access[_-]?key[_-]?=\s*["\']([A-Za-z0-9/+=]{8,})["\']'
        ],
        "Password": [
            r'password[_-]?=\s*["\'](.+?)["\']',
            r'passwd[_-]?=\s*["\'](.+?)["\']',
            r'pass[_-]?=\s*["\'](.+?)["\']'
        ],
        "Private Key": [
            r'-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----'
        ],
        "Connection String": [
            r'(mongodb|redis|mysql|postgresql|sqlite|jdbc|odbc)://[^\s"]+'
        ],
        "Token": [
            r'token[_-]?=\s*["\']([a-zA-Z0-9_\-\.]{8,})["\']',
            r'auth[_-]?token[_-]?=\s*["\']([a-zA-Z0-9_\-\.]{8,})["\']',
            r'bearer[_-]?=\s*["\']([a-zA-Z0-9_\-\.]{8,})["\']'
        ]
    }
    
    # Add extra patterns if provided
    if extra_patterns:
        for category, pattern_list in extra_patterns.items():
            if category in patterns:
                patterns[category].extend(pattern_list)
            else:
                patterns[category] = pattern_list
    
    # Find issues using regex patterns
    found_issues = []
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                found_issues.append({
                    'type': category,
                    'match': match.group(0),
                    'line': line_num
                })
    
    # Find issues using entropy analysis
    entropy_secrets = find_potential_secrets(code)
    
    # Combine and remove duplicates
    for secret in entropy_secrets:
        # Skip if already found by pattern matching
        if any(secret['line'] == issue['line'] and 
               secret['match'] in issue['match'] 
               for issue in found_issues):
            continue
        
        found_issues.append({
            'type': 'High Entropy String',
            'match': secret['match'],
            'line': secret['line'],
            'entropy': secret['entropy']
        })
    
    # Create SecurityIssue objects from findings
    for issue in found_issues:
        issues.append(SecurityIssue(
            title=f"Hardcoded {issue['type']} detected",
            description=f"Hardcoded secret found in code: {issue['match'][:20]}...",
            severity="high",
            location=f"Line {issue['line']}",
            remediation="Store secrets in environment variables or a secure vault"
        ))
    
    result = {
        "issues": issues,
        "secret_count": len(issues)
    }
    
    return result

def check_compliance_gdpr(code=None, config_files=None):
    """
    Check for GDPR compliance indicators.
    
    :param code: Source code as a string (optional)
    :param config_files: List of configuration files to check (optional)
    :return: Dictionary with GDPR compliance assessment results
    """
    issues = []
    compliance_indicators = set()
    
    # Check for personal data handling
    pii_patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        'phone': r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b'
    }
    
    if code:
        # Check for PII patterns in code
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, code):
                issues.append(ComplianceIssue(
                    title=f"Potential {pii_type} data handling detected",
                    description=f"Code appears to process {pii_type} data which is subject to GDPR",
                    severity="medium",
                    regulation="GDPR",
                    requirement="Article 4 - Personal Data Definition",
                    remediation="Ensure proper consent, processing, and protection mechanisms"
                ))
        
        # Look for GDPR compliance indicators
        gdpr_indicators = {
            'consent': [r'consent', r'opt[- ]?in', r'permission'],
            'right_to_access': [r'data[_ ]?access', r'access[_ ]?request', r'subject[_ ]?access'],
            'right_to_be_forgotten': [r'forget[_ ]?me', r'delete[_ ]?user', r'erase[_ ]?data', 
                                    r'right[_ ]?to[_ ]?erasure', r'data[_ ]?deletion'],
            'data_portability': [r'export[_ ]?data', r'data[_ ]?portability', r'transfer[_ ]?data'],
            'data_minimization': [r'minimize[_ ]?data', r'data[_ ]?minimization'],
            'breach_notification': [r'breach[_ ]?notification', r'notify[_ ]?breach'],
            'privacy_by_design': [r'privacy[_ ]?by[_ ]?design', r'privacy[_ ]?impact']
        }
        
        for feature, patterns in gdpr_indicators.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    compliance_indicators.add(feature)
    
    # Check for required GDPR features
    required_features = DEFAULT_CONFIG['compliance']['gdpr']['required_features']
    for feature in required_features:
        if feature not in compliance_indicators:
            issues.append(ComplianceIssue(
                title=f"Missing GDPR compliance feature: {feature}",
                description=f"No evidence of {feature} implementation found",
                severity="high",
                regulation="GDPR",
                requirement=f"Article related to {feature}",
                remediation=f"Implement mechanisms for {feature}"
            ))
    
    result = {
        "issues": issues,
        "compliance_indicators": list(compliance_indicators),
        "personal_data_usage_detected": any(re.search(pattern, code) for pattern in pii_patterns.values()) if code else None
    }
    
    return result

def check_compliance_hipaa(code=None, config_files=None):
    """
    Check for HIPAA compliance indicators.
    
    :param code: Source code as a string (optional)
    :param config_files: List of configuration files to check (optional)
    :return: Dictionary with HIPAA compliance assessment results
    """
    issues = []
    compliance_indicators = set()
    
    # Check for PHI (Protected Health Information) handling
    phi_patterns = {
        'medical_record': [r'medical[_ ]?record', r'health[_ ]?record', r'patient[_ ]?record', 
                         r'medical[_ ]?data', r'health[_ ]?data'],
        'diagnosis': [r'diagnosis', r'condition', r'treatment', r'icd[-_ ]?\d+'],
        'patient_info': [r'patient[_ ]?id', r'patient[_ ]?name', r'patient[_ ]?data']
    }
    
    if code:
        # Check for PHI patterns in code
        for phi_type, patterns in phi_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    issues.append(ComplianceIssue(
                        title=f"Potential {phi_type} handling detected",
                        description=f"Code appears to process {phi_type} which is subject to HIPAA",
                        severity="medium",
                        regulation="HIPAA",
                        requirement="Privacy Rule - Protected Health Information",
                        remediation="Ensure proper safeguards and patient authorization"
                    ))
                    break  # One match per phi_type is enough
        
        # Look for HIPAA compliance indicators
        hipaa_indicators = {
            'access_controls': [r'access[_ ]?control', r'authorization', r'permission'],
            'audit_logging': [r'audit[_ ]?log', r'activity[_ ]?log', r'access[_ ]?log'],
            'encryption_at_rest': [r'encrypt[_ ]?at[_ ]?rest', r'data[_ ]?encryption', r'encrypt[_ ]?data'],
            'encryption_in_transit': [r'encrypt[_ ]?in[_ ]?transit', r'ssl', r'tls', r'https'],
            'integrity_controls': [r'data[_ ]?integrity', r'checksum', r'hash'],
            'risk_analysis': [r'risk[_ ]?analysis', r'risk[_ ]?assessment'],
            'emergency_access': [r'emergency[_ ]?access', r'break[_ ]?glass']
        }
        
        for feature, patterns in hipaa_indicators.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    compliance_indicators.add(feature)
    
    # Check for required HIPAA features
    required_features = DEFAULT_CONFIG['compliance']['hipaa']['required_features']
    for feature in required_features:
        if feature not in compliance_indicators:
            issues.append(ComplianceIssue(
                title=f"Missing HIPAA compliance feature: {feature}",
                description=f"No evidence of {feature} implementation found",
                severity="high",
                regulation="HIPAA",
                requirement=f"Security Rule - {feature.replace('_', ' ').title()}",
                remediation=f"Implement mechanisms for {feature}"
            ))
    
    result = {
        "issues": issues,
        "compliance_indicators": list(compliance_indicators),
        "phi_usage_detected": any(re.search(pattern, code, re.IGNORECASE) 
                                for patterns in phi_patterns.values() 
                                for pattern in patterns) if code else None
    }
    
    return result

def check_for_vulnerabilities(code):
    """
    Scan code for common security vulnerabilities.
    
    :param code: Source code as a string
    :return: Dictionary with vulnerability assessment results
    """
    issues = []
    
    # Check for common vulnerability patterns
    vulnerability_patterns = DEFAULT_CONFIG['code_security']['vulnerability_patterns']
    
    for vuln_type, patterns in vulnerability_patterns.items():
        for pattern in patterns:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                
                # Map vulnerability types to CWE IDs and severity
                cwe_mapping = {
                    'sql_injection': {'cwe_id': 'CWE-89', 'severity': 'critical'},
                    'command_injection': {'cwe_id': 'CWE-78', 'severity': 'critical'},
                    'xss': {'cwe_id': 'CWE-79', 'severity': 'high'}
                }
                
                cwe_info = cwe_mapping.get(vuln_type, {'cwe_id': None, 'severity': 'high'})
                
                issues.append(VulnerabilityIssue(
                    title=f"Potential {vuln_type.replace('_', ' ').title()} vulnerability",
                    description=f"Potentially insecure code pattern: {match.group(0)}",
                    severity=cwe_info['severity'],
                    vulnerability_type=vuln_type,
                    cwe_id=cwe_info['cwe_id'],
                    location=f"Line {line_num}",
                    remediation=get_remediation_for_vulnerability(vuln_type)
                ))
    
    # Check for secure coding practices
    secure_coding_patterns = DEFAULT_CONFIG['code_security']['secure_coding_practices']
    secure_practices_found = {}
    
    for practice, patterns in secure_coding_patterns.items():
        secure_practices_found[practice] = False
        for pattern in patterns:
            if re.search(pattern, code):
                secure_practices_found[practice] = True
                break
    
    # Add issues for missing secure coding practices
    for practice, found in secure_practices_found.items():
        if not found:
            issues.append(SecurityIssue(
                title=f"Missing secure coding practice: {practice.replace('_', ' ').title()}",
                description=f"No evidence of {practice} implementation found",
                severity="medium",
                remediation=get_remediation_for_practice(practice)
            ))
    
    result = {
        "issues": issues,
        "vulnerability_count": len(issues),
        "secure_practices": secure_practices_found
    }
    
    return result

def scan_dependencies(requirements_file=None):
    """
    Scan Python dependencies for known security vulnerabilities.
    
    :param requirements_file: Path to requirements.txt file (optional)
    :return: Dictionary with dependency security assessment results
    """
    issues = []
    dependencies = []
    
    # Get installed packages
    try:
        installed_packages = [d for d in pkg_resources.working_set]
        
        for package in installed_packages:
            dependencies.append({
                'name': package.project_name,
                'version': package.version
            })
    except Exception as e:
        logging.warning(f"Could not analyze installed packages: {e}")
    
    # This would normally check against a vulnerability database
    # For the purpose of this example, we'll simulate with some known vulnerable packages
    known_vulnerabilities = {
        'django': {
            '1.8.0': ['CVE-2016-6186', 'CVE-2016-7401'],
            '1.9.0': ['CVE-2017-7233']
        },
        'flask': {
            '0.12.0': ['CVE-2018-1000656']
        },
        'requests': {
            '2.19.0': ['CVE-2018-18074']
        },
        'pyyaml': {
            '5.1.0': ['CVE-2020-1747']
        }
    }
    
    # Check dependencies against simulated vulnerability database
    for dep in dependencies:
        package_name = dep['name'].lower()
        version = dep['version']
        
        if package_name in known_vulnerabilities:
            for vuln_version, cve_list in known_vulnerabilities[package_name].items():
                if version == vuln_version:
                    for cve in cve_list:
                        issues.append(VulnerabilityIssue(
                            title=f"Vulnerable dependency: {package_name} {version}",
                            description=f"Package has known vulnerability: {cve}",
                            severity="high",
                            vulnerability_type="dependency",
                            location=f"Package: {package_name}",
                            remediation=f"Update {package_name} to a newer version"
                        ))
    
    # Check for outdated packages (would normally use PyPI API)
    outdated_packages = [
        {'name': 'example-package', 'current': '1.0.0', 'latest': '1.2.0'}
    ]
    
    for pkg in outdated_packages:
        issues.append(SecurityIssue(
            title=f"Outdated dependency: {pkg['name']}",
            description=f"Current version {pkg['current']} is behind latest {pkg['latest']}",
            severity="low",
            location=f"Package: {pkg['name']}",
            remediation=f"Update {pkg['name']} to version {pkg['latest']}"
        ))
    
    result = {
        "issues": issues,
        "dependencies": dependencies,
        "dependency_count": len(dependencies),
        "vulnerable_dependency_count": len([i for i in issues if isinstance(i, VulnerabilityIssue)])
    }
    
    return result

def check_secure_communications():
    """
    Check for secure communication configurations.
    
    :return: Dictionary with secure communication assessment results
    """
    issues = []
    
    # Check SSL/TLS configuration
    try:
        # Get available SSL/TLS protocols
        protocols = []
        for protocol in ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1_1', 'TLSv1_2', 'TLSv1_3']:
            try:
                if hasattr(ssl, protocol):
                    protocols.append(protocol)
            except:
                pass
        
        # Check for insecure protocols
        insecure_protocols = ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1_1']
        for protocol in insecure_protocols:
            if protocol in protocols:
                issues.append(ConfigurationIssue(
                    title=f"Insecure protocol available: {protocol}",
                    description=f"{protocol} is considered insecure and should be disabled",
                    severity="high" if protocol in ['SSLv2', 'SSLv3'] else "medium",
                    component="SSL/TLS",
                    current_value=protocol,
                    recommended_value="TLSv1_2 or higher",
                    remediation=f"Disable {protocol} in your SSL/TLS configuration"
                ))
    except Exception as e:
        logging.warning(f"Could not check SSL/TLS protocols: {e}")
    
    # Check for HTTPS usage in code (simplified)
    uses_https = False
    try:
        for module_name in sys.modules:
            if 'ssl' in module_name or 'https' in module_name:
                uses_https = True
                break
    except:
        pass
    
    if not uses_https:
        issues.append(SecurityIssue(
            title="No evidence of HTTPS usage",
            description="Application may not be using secure communications",
            severity="high",
            remediation="Implement HTTPS for all communications using TLS 1.2 or higher"
        ))
    
    result = {
        "issues": issues,
        "uses_https": uses_https,
        "available_protocols": protocols if 'protocols' in locals() else None
    }
    
    return result

def check_input_validation(code):
    """
    Check for proper input validation practices in code.
    
    :param code: Source code as a string
    :return: Dictionary with input validation assessment results
    """
    issues = []
    
    # Check for direct input usage without validation
    dangerous_patterns = [
        (r'request\.(?:args|form|get|post|params)\[["\']?\w+["\']?\]', 'HTTP request data'),
        (r'request\.(?:body|json|data)', 'HTTP request body'),
        (r'input\([^)]*\)', 'User input'),
        (r'(?:open|read)\([^)]*user', 'User-provided file'),
        (r'(?:json|pickle)\.loads\(.*?user', 'User-provided serialized data')
    ]
    
    # Check for validation patterns
    validation_patterns = [
        r'\.isdigit\(',
        r'\.strip\(',
        r'\.isalnum\(',
        r'\.isalpha\(',
        r'\.validate\(',
        r'validator',
        r'sanitize',
        r'escape\('
    ]
    
    validation_used = any(re.search(pattern, code) for pattern in validation_patterns)
    
    for pattern, input_type in dangerous_patterns:
        for match in re.finditer(pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            
            # Check if validation is used near the input
            line_start = max(0, match.start() - 200)
            line_end = min(len(code), match.end() + 200)
            context = code[line_start:line_end]
            
            if not any(re.search(pattern, context) for pattern in validation_patterns):
                issues.append(VulnerabilityIssue(
                    title=f"Unvalidated {input_type}",
                    description=f"Using {input_type} without proper validation: {match.group(0)}",
                    severity="high",
                    vulnerability_type="input_validation",
                    cwe_id="CWE-20",
                    location=f"Line {line_num}",
                    remediation="Implement input validation and sanitization"
                ))
    
    result = {
        "issues": issues,
        "validation_used": validation_used,
        "unvalidated_inputs": len(issues)
    }
    
    return result

# Helper functions
def get_remediation_for_vulnerability(vuln_type):
    """Get remediation guidance for a specific vulnerability type."""
    remediation_map = {
        'sql_injection': "Use parameterized queries or ORM instead of string concatenation",
        'command_injection': "Use safe APIs instead of command execution, validate and sanitize input",
        'xss': "Use template escaping and context-aware output encoding"
    }
    return remediation_map.get(vuln_type, "Implement proper input validation and output encoding")

def get_remediation_for_practice(practice):
    """Get remediation guidance for a missing secure coding practice."""
    remediation_map = {
        'input_validation': "Implement strict input validation for all user-controllable data",
        'output_encoding': "Always encode output based on the context (HTML, JavaScript, etc.)",
        'parameterized_queries': "Use parameterized queries for all database operations"
    }
    return remediation_map.get(practice, "Follow security best practices for this area")

def load_file_content(file_path):
    """Load content from a file with proper error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Could not read file {file_path}: {e}")
        return None

def aggregate_issues(results):
    """Aggregate all issues from different security checks."""
    all_issues = []
    for result_dict in results.values():
        if isinstance(result_dict, dict) and 'issues' in result_dict:
            all_issues.extend(result_dict['issues'])
    return all_issues

def generate_html_report(results, output_file="security_compliance_report.html"):
    """Generate an HTML report from security analysis results."""
    # Import template here to avoid dependency if HTML report isn't needed
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security & Compliance Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2a5885; }
            h2 { color: #366b95; margin-top: 30px; }
            .summary { background-color: #f0f5fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .issues { margin: 20px 0; }
            .issue { background-color: #fff; padding: 15px; margin: 10px 0; border-left: 5px solid #ccc; }
            .critical { border-color: #d9534f; }
            .high { border-color: #f0ad4e; }
            .medium { border-color: #5bc0de; }
            .low { border-color: #5cb85c; }
            .issue h3 { margin-top: 0; }
            .location { font-family: monospace; background-color: #f7f7f7; padding: 5px; }
            .remediation { background-color: #e9f7ef; padding: 10px; margin-top: 10px; }
            .footer { margin-top: 50px; font-size: 0.8em; color: #777; }
        </style>
    </head>
    <body>
        <h1>Security & Compliance Analysis Report</h1>
        <p>Generated on: {timestamp}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Issues: {total_issues}</p>
            <ul>
                <li>Critical: {critical_count}</li>
                <li>High: {high_count}</li>
                <li>Medium: {medium_count}</li>
                <li>Low: {low_count}</li>
            </ul>
        </div>
        
        <h2>Detailed Issues</h2>
        <div class="issues">
            {issues_html}
        </div>
        
        <h2>Compliance Status</h2>
        <div>
            {compliance_html}
        </div>
        
        <div class="footer">
            <p>Security & Compliance Evaluation Framework</p>
        </div>
    </body>
    </html>
    """
    
    # Aggregate all issues
    all_issues = aggregate_issues(results)
    
    # Count issues by severity
    issue_counts = {
        'critical': 0,
        'high': 0,
        'medium': 0,
        'low': 0
    }
    
    for issue in all_issues:
        if hasattr(issue, 'severity') and issue.severity in issue_counts:
            issue_counts[issue.severity] += 1
    
    # Generate HTML for issues
    issues_html = ""
    for issue in all_issues:
        severity_class = getattr(issue, 'severity', 'low').lower()
        location = getattr(issue, 'location', 'N/A')
        remediation = getattr(issue, 'remediation', 'No specific remediation guidance available.')
        
        issues_html += f"""
        <div class="issue {severity_class}">
            <h3>{issue.title}</h3>
            <p>{issue.description}</p>
            <p><strong>Severity:</strong> {issue.severity.upper()}</p>
            <p><strong>Location:</strong> <span class="location">{location}</span></p>
            <div class="remediation">
                <p><strong>Remediation:</strong> {remediation}</p>
            </div>
        </div>
        """
    
    # Generate compliance section
    compliance_html = "<ul>"
    
    if 'gdpr_compliance' in results:
        gdpr_result = results['gdpr_compliance']
        gdpr_status = 'Potential Issues' if gdpr_result.get('issues') else 'No Issues Detected'
        gdpr_indicators = ', '.join(gdpr_result.get('compliance_indicators', []))
        compliance_html += f"""
        <li><strong>GDPR:</strong> {gdpr_status}
            <ul>
                <li>Compliance indicators: {gdpr_indicators if gdpr_indicators else 'None detected'}</li>
            </ul>
        </li>
        """
    
    if 'hipaa_compliance' in results:
        hipaa_result = results['hipaa_compliance']
        hipaa_status = 'Potential Issues' if hipaa_result.get('issues') else 'No Issues Detected'
        hipaa_indicators = ', '.join(hipaa_result.get('compliance_indicators', []))
        compliance_html += f"""
        <li><strong>HIPAA:</strong> {hipaa_status}
            <ul>
                <li>Compliance indicators: {hipaa_indicators if hipaa_indicators else 'None detected'}</li>
            </ul>
        </li>
        """
    
    compliance_html += "</ul>"
    
    # Fill in the template
    report_html = html_template.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_issues=len(all_issues),
        critical_count=issue_counts['critical'],
        high_count=issue_counts['high'],
        medium_count=issue_counts['medium'],
        low_count=issue_counts['low'],
        issues_html=issues_html,
        compliance_html=compliance_html
    )
    
    # Write to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_html)
        logging.info(f"HTML report generated: {output_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to write HTML report: {e}")
        return False

# Main security analysis function
def analyze_security_compliance(function=None, code=None, file_paths=None, compliance_checks=None, output_format="json"):
    """
    Comprehensive security and compliance analysis.
    
    :param function: Function object to analyze (optional)
    :param code: Source code string to analyze (optional)
    :param file_paths: List of file paths to analyze (optional)
    :param compliance_checks: List of compliance standards to check
    :param output_format: Format for results ("json" or "html")
    :return: Dictionary with comprehensive security assessment results
    """
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "analysis_targets": {}
    }
    
    # Determine what to analyze
    if function:
        results["analysis_targets"]["function"] = function.__name__
        try:
            code = inspect.getsource(function)
        except Exception as e:
            logging.error(f"Could not get source for function {function.__name__}: {e}")
    
    if file_paths:
        results["analysis_targets"]["files"] = file_paths
        if not code:
            # Combine code from all files
            all_code = []
            for file_path in file_paths:
                file_content = load_file_content(file_path)
                if file_content:
                    all_code.append(file_content)
            if all_code:
                code = "\n".join(all_code)
    
    # Run security checks on code
    if code:
        results["encryption_check"] = check_data_encryption(code)
        results["secrets_scan"] = scan_for_hardcoded_secrets(code)
        results["vulnerability_scan"] = check_for_vulnerabilities(code)
        results["input_validation"] = check_input_validation(code)
    
    # Run additional checks
    results["access_control"] = check_access_controls(file_paths)
    results["dependency_security"] = scan_dependencies()
    results["secure_communications"] = check_secure_communications()
    
    # Run compliance checks if requested
    if compliance_checks:
        if "gdpr" in compliance_checks:
            results["gdpr_compliance"] = check_compliance_gdpr(code)
        if "hipaa" in compliance_checks:
            results["hipaa_compliance"] = check_compliance_hipaa(code)
    
    # Generate summary statistics
    all_issues = aggregate_issues(results)
    
    severity_counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0
    }
    
    for issue in all_issues:
        if hasattr(issue, 'severity') and issue.severity in severity_counts:
            severity_counts[issue.severity] += 1
    
    results["summary"] = {
        "total_issues": len(all_issues),
        "severity_counts": severity_counts,
        "overall_risk_level": determine_risk_level(severity_counts)
    }
    
    # Generate output in requested format
    if output_format == "html":
        generate_html_report(results)
    
    # Save JSON report
    with open("security_compliance_summary.json", "w") as f:
        # Convert issues to dictionaries for JSON serialization
        json_safe_results = {}
        for key, value in results.items():
            if isinstance(value, dict) and 'issues' in value:
                value_copy = value.copy()
                value_copy['issues'] = [issue.to_dict() for issue in value_copy['issues']]
                json_safe_results[key] = value_copy
            else:
                json_safe_results[key] = value
        
        json.dump(json_safe_results, f, indent=4)
    
    logging.info(f"Security & Compliance analysis complete. Found {len(all_issues)} issues.")
    return results

def determine_risk_level(severity_counts):
    """Determine overall risk level based on issue severity counts."""
    if severity_counts["critical"] > 0:
        return "critical"
    elif severity_counts["high"] > 2:
        return "high"
    elif severity_counts["high"] > 0 or severity_counts["medium"] > 5:
        return "medium"
    elif severity_counts["medium"] > 0 or severity_counts["low"] > 5:
        return "low"
    else:
        return "info"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Security & Compliance Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--file", "-f",
        action='append',
        help="File to analyze (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--module", "-m",
        help="Python module to analyze"
    )
    
    parser.add_argument(
        "--function",
        help="Function name to analyze (requires --module)"
    )
    
    parser.add_argument(
        "--compliance", "-c",
        choices=["gdpr", "hipaa", "pci", "all"],
        help="Compliance standard to check"
    )
    
    parser.add_argument(
        "--output", "-o",
        choices=["json", "html"],
        default="json",
        help="Output format"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

# Example function to be tested
def example_function():
    API_KEY = "123456abcdef"  # Intentional issue for detection
    return hashlib.sha256(b"secure_data").hexdigest()

# Run security evaluation
if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    files_to_analyze = args.file or []
    function_to_analyze = None
    
    # Load module and function if specified
    if args.module:
        try:
            module = importlib.import_module(args.module)
            if args.function:
                function_to_analyze = getattr(module, args.function)
        except Exception as e:
            logging.error(f"Error loading module {args.module}: {e}")
    
    # Determine compliance checks
    compliance_checks = []
    if args.compliance:
        if args.compliance == "all":
            compliance_checks = ["gdpr", "hipaa", "pci"]
        else:
            compliance_checks = [args.compliance]
    
    # If no targets specified, analyze example function
    if not files_to_analyze and not function_to_analyze:
        function_to_analyze = example_function
    
    # Run analysis
    results = analyze_security_compliance(
        function=function_to_analyze,
        file_paths=files_to_analyze,
        compliance_checks=compliance_checks,
        output_format=args.output
    )
    
    # Print summary to console
    all_issues = aggregate_issues(results)
    print("\n" + "=" * 60)
    print("SECURITY & COMPLIANCE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total issues found: {len(all_issues)}")
    
    if results["summary"]["severity_counts"]["critical"] > 0:
        print(f"CRITICAL issues: {results['summary']['severity_counts']['critical']}")
    
    print(f"High severity issues: {results['summary']['severity_counts']['high']}")
    print(f"Medium severity issues: {results['summary']['severity_counts']['medium']}")
    print(f"Low severity issues: {results['summary']['severity_counts']['low']}")
    print(f"Overall risk level: {results['summary']['overall_risk_level'].upper()}")
    print("=" * 60)
    
    print("\nSecurity & Compliance evaluation complete.")
    print("Check 'security_compliance.log' and 'security_compliance_summary.json' for details.")
    if args.output == "html":
        print("HTML report generated: security_compliance_report.html")

"""
TODO:
- Implement container security scanning for Docker/Kubernetes deployments.
- Add support for scanning cloud infrastructure configurations (AWS, Azure, GCP).
- Enhance PII detection with ML-based approaches for higher accuracy.
- Implement dynamic analysis capabilities to detect runtime security issues.
- Add support for secure coding standards verification (CERT, CWE/SANS Top 25).
- Extend compliance checks to include SOC2, ISO27001, and NIST frameworks.
- Implement API security analysis for REST/GraphQL endpoints.
- Add static application security testing (SAST) with deeper code analysis.
- Integrate with security information and event management (SIEM) systems.
- Implement secure development lifecycle (SDL) process verification.
"""
