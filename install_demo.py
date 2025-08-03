#!/usr/bin/env python3
"""
Simple installation script for demo purposes

This script installs only the essential dependencies needed to run the application demo.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package):
    """Check if a package is installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    print("🚀 Installing essential dependencies for Demand Forecasting API Demo")
    print("=" * 60)
    
    # Essential packages for the demo
    essential_packages = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0", 
        "pydantic>=2.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "python-multipart>=0.0.6",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-dotenv>=1.0.0",
        "structlog>=23.2.0",
        "prometheus-client>=0.19.0"
    ]
    
    installed = []
    failed = []
    
    for package in essential_packages:
        package_name = package.split(">=")[0].split("[")[0]
        print(f"📦 Installing {package_name}...")
        
        if install_package(package):
            installed.append(package_name)
            print(f"✅ {package_name} installed successfully")
        else:
            failed.append(package_name)
            print(f"❌ Failed to install {package_name}")
    
    print("\n" + "=" * 60)
    print("📊 Installation Summary:")
    print(f"✅ Successfully installed: {len(installed)} packages")
    
    if installed:
        print("   - " + "\n   - ".join(installed))
    
    if failed:
        print(f"❌ Failed to install: {len(failed)} packages")
        print("   - " + "\n   - ".join(failed))
        print("\nNote: Some packages might require system dependencies.")
        print("For a full installation, use: pip install -r requirements.txt")
    
    print("\n🎉 Basic installation completed!")
    print("You can now run: python3 main.py --mode api")
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()