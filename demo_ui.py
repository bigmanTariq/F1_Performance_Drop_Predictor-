#!/usr/bin/env python3
"""
F1 Performance Drop Predictor - UI Demo

This script demonstrates the new web UI for the F1 Performance Drop Predictor.
It starts the server and provides instructions for accessing the beautiful web interface.
"""

import subprocess
import time
import webbrowser
import sys
import os
from pathlib import Path

def check_server_status():
    """Check if the API server is running"""
    try:
        import requests
        response = requests.get('http://localhost:8000/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the API server"""
    print("🚀 Starting F1 Performance Drop Predictor API server...")
    
    # Start server in background
    process = subprocess.Popen([
        sys.executable, "src/serve.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("⏳ Waiting for server to initialize...")
    for i in range(30):  # Wait up to 30 seconds
        if check_server_status():
            print("✅ Server is running!")
            return process
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}s)")
    
    print("❌ Server failed to start")
    return None

def main():
    """Main demo function"""
    print("=" * 60)
    print("🏎️  F1 PERFORMANCE DROP PREDICTOR - WEB UI DEMO")
    print("=" * 60)
    print()
    
    # Check if static files exist
    static_dir = Path("static")
    if not static_dir.exists():
        print("❌ Static files directory not found!")
        print("   Make sure you're running this from the project root directory.")
        return
    
    required_files = ["index.html", "styles.css", "app.js"]
    missing_files = [f for f in required_files if not (static_dir / f).exists()]
    
    if missing_files:
        print(f"❌ Missing static files: {missing_files}")
        return
    
    print("📁 Static files found ✅")
    
    # Start server if not running
    if not check_server_status():
        server_process = start_server()
        if not server_process:
            return
    else:
        print("✅ Server is already running!")
        server_process = None
    
    print()
    print("🌐 WEB UI ACCESS INFORMATION")
    print("-" * 40)
    print("📱 Main UI:           http://localhost:8000/ui")
    print("📊 API Docs:          http://localhost:8000/docs")
    print("🔧 Health Check:      http://localhost:8000/health")
    print("📋 Model Info:        http://localhost:8000/model_info")
    print()
    
    print("🎨 UI FEATURES")
    print("-" * 40)
    print("✨ Beautiful F1-themed interface with dark mode")
    print("🏁 Pre-built race scenarios (Championship Leader, Midfield, Backmarker)")
    print("⚙️  Interactive form with 47 race parameters")
    print("📈 Real-time prediction results with visualizations")
    print("🎯 Probability gauge and position change forecasts")
    print("📊 Feature importance analysis")
    print("📱 Responsive design for mobile and desktop")
    print()
    
    print("🚀 QUICK START SCENARIOS")
    print("-" * 40)
    print("1. 👑 Championship Leader - High pressure, pole position")
    print("2. ⚔️  Midfield Battle - Competitive racing, street circuit")
    print("3. 🏃 Backmarker Team - Development focus, reliability issues")
    print("4. 🛠️  Custom Scenario - Build your own race situation")
    print()
    
    # Try to open browser
    try:
        print("🌐 Opening web browser...")
        webbrowser.open('http://localhost:8000/ui')
        print("✅ Browser opened! If it didn't open automatically, visit:")
        print("   👉 http://localhost:8000/ui")
    except:
        print("⚠️  Could not open browser automatically. Please visit:")
        print("   👉 http://localhost:8000/ui")
    
    print()
    print("💡 USAGE TIPS")
    print("-" * 40)
    print("• Start with a pre-built scenario, then customize parameters")
    print("• Use 'Show Advanced Parameters' for full control")
    print("• Watch the probability gauge change as you adjust values")
    print("• Check feature importance to understand key factors")
    print("• Try extreme scenarios to see model behavior")
    print()
    
    print("🛑 TO STOP THE SERVER")
    print("-" * 40)
    print("Press Ctrl+C in this terminal or run:")
    print("pkill -f 'python src/serve.py'")
    print()
    
    if server_process:
        try:
            print("Press Ctrl+C to stop the server...")
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Server stopped!")

if __name__ == "__main__":
    main()