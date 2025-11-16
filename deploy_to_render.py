#!/usr/bin/env python3
"""
Deployment helper script for Render.com
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main deployment function."""
    print("🚀 F1 Performance Drop Predictor - Render Deployment Helper")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/serve.py"):
        print("❌ Error: Must run from project root directory")
        sys.exit(1)
    
    # Run local tests first
    print("🧪 Running local tests...")
    if not run_command("python test_local_deployment.py", "Local deployment test"):
        print("❌ Local tests failed. Fix issues before deploying.")
        sys.exit(1)
    
    # Check git status
    print("\n📋 Checking git status...")
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("📝 Uncommitted changes found:")
        print(result.stdout)
        
        commit = input("Commit changes? (y/n): ").lower().strip()
        if commit == 'y':
            commit_msg = input("Commit message (or press Enter for default): ").strip()
            if not commit_msg:
                commit_msg = "Update deployment configuration"
            
            if not run_command("git add .", "Adding files"):
                sys.exit(1)
            if not run_command(f'git commit -m "{commit_msg}"', "Committing changes"):
                sys.exit(1)
        else:
            print("⚠️ Warning: Deploying with uncommitted changes")
    
    # Push to git
    print("\n📤 Pushing to git...")
    if not run_command("git push", "Pushing to repository"):
        print("❌ Git push failed. Check your repository setup.")
        sys.exit(1)
    
    print("\n🎉 Deployment preparation complete!")
    print("\nNext steps:")
    print("1. Go to your Render dashboard: https://dashboard.render.com/")
    print("2. Your service should automatically redeploy from the git push")
    print("3. Monitor the build logs for any issues")
    print("4. Test the health endpoint once deployed:")
    print("   https://f1-performance-drop-predictor.onrender.com/health")
    print("\n📚 If you encounter issues, check DEPLOYMENT_TROUBLESHOOTING.md")

if __name__ == "__main__":
    main()