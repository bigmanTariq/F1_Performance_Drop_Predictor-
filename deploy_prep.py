#!/usr/bin/env python3
"""
Deployment preparation script for F1 Performance Drop Predictor.
This script helps you get everything ready for cloud deployment.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"   ✅ {description}")
        return True
    else:
        print(f"   ❌ Missing: {description}")
        return False

def main():
    """Main deployment preparation function."""
    print("🏎️ F1 Performance Drop Predictor - Deployment Preparation")
    print("=" * 60)
    
    success_count = 0
    total_checks = 0
    
    # Check 1: Required files
    print("\n1️⃣ Checking deployment files...")
    files_to_check = [
        ("render.yaml", "Render.com configuration"),
        ("build.sh", "Build script"),
        ("runtime.txt", "Python runtime specification"),
        ("Procfile", "Process file"),
        ("requirements.txt", "Python dependencies"),
        ("src/serve.py", "FastAPI server"),
        ("src/data_prep.py", "Data preparation script"),
        ("src/train.py", "Model training script"),
        ("DEPLOYMENT.md", "Deployment documentation")
    ]
    
    for filepath, description in files_to_check:
        total_checks += 1
        if check_file_exists(filepath, description):
            success_count += 1
    
    # Check 2: Build script permissions
    print("\n2️⃣ Checking file permissions...")
    total_checks += 1
    if run_command("chmod +x build.sh", "Making build script executable"):
        success_count += 1
    
    # Check 3: Test data preparation
    print("\n3️⃣ Testing data preparation...")
    total_checks += 1
    if run_command("python src/data_prep.py", "Running data preparation"):
        success_count += 1
    
    # Check 4: Test model training
    print("\n4️⃣ Testing model training...")
    total_checks += 1
    if run_command("python src/train.py", "Training models"):
        success_count += 1
    
    # Check 5: Test local server
    print("\n5️⃣ Testing local server startup...")
    print("   🔧 Starting server (will stop after 5 seconds)...")
    try:
        # Start server in background and test health endpoint
        server_process = subprocess.Popen(
            ["python", "src/serve.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        import time
        time.sleep(3)
        
        # Test health endpoint
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Server starts and responds correctly")
                success_count += 1
            else:
                print(f"   ❌ Server responded with status {response.status_code}")
        except Exception as e:
            print(f"   ❌ Could not connect to server: {str(e)}")
        
        # Stop the server
        server_process.terminate()
        server_process.wait()
        total_checks += 1
        
    except Exception as e:
        print(f"   ❌ Could not start server: {str(e)}")
        total_checks += 1
    
    # Check 6: Git status
    print("\n6️⃣ Checking Git status...")
    total_checks += 1
    if run_command("git status --porcelain", "Checking for uncommitted changes"):
        # If git status returns empty (no uncommitted changes), that's good
        result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if not result.stdout.strip():
            print("   ✅ All changes committed")
            success_count += 1
        else:
            print("   ⚠️ You have uncommitted changes:")
            print(f"   {result.stdout}")
            print("   💡 Run: git add . && git commit -m 'Prepare for deployment'")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 DEPLOYMENT READINESS: {success_count}/{total_checks} checks passed")
    
    if success_count == total_checks:
        print("🎉 READY FOR DEPLOYMENT!")
        print("\n🚀 Next steps:")
        print("1. Push to GitHub: git push origin main")
        print("2. Go to render.com and create a new web service")
        print("3. Connect your GitHub repository")
        print("4. Use the settings from DEPLOYMENT.md")
        print("5. Deploy and test with: python scripts/test_deployment.py <your-url>")
        
    elif success_count >= total_checks * 0.8:
        print("✅ MOSTLY READY - Minor issues to fix")
        print("\n🔧 Fix the issues above, then you're ready to deploy!")
        
    else:
        print("❌ NOT READY - Several issues need attention")
        print("\n🛠️ Please fix the failed checks before deploying")
    
    print(f"\n📖 See DEPLOYMENT.md for detailed deployment instructions")
    print(f"📋 See DEPLOYMENT_CHECKLIST.md for step-by-step checklist")

if __name__ == "__main__":
    main()