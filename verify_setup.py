"""
Quick verification script to check if the system is ready to run.
"""
import sys

def check_imports():
    """Check if all required packages are installed."""
    required = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'numpy': 'NumPy',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'speechbrain': 'SpeechBrain',
        'sklearn': 'scikit-learn',
        'whisper': 'OpenAI Whisper'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

def check_models():
    """Check if required models are downloaded."""
    import os
    
    # Check pretrained models
    model_paths = [
        'pretrained_models/spkrec',
        'pretrained_models/asr'
    ]
    
    missing = []
    for path in model_paths:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print(f"⚠️  Missing model directories: {', '.join(missing)}")
        print("\nDownload models with: python download_models.py")
        return False
    
    print("✅ Model directories exist")
    return True

def check_torch_compatibility():
    """Check for torch/torchaudio compatibility issues."""
    try:
        import torch
        import torchaudio
        
        # Try to load torchaudio backend
        try:
            torchaudio.set_audio_backend("soundfile")
            print("✅ TorchAudio backend: soundfile")
        except Exception as e:
            print(f"⚠️  TorchAudio backend warning: {e}")
            print("   System will still work with default backend")
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ TorchAudio version: {torchaudio.__version__}")
        return True
        
    except Exception as e:
        print(f"❌ Torch compatibility issue: {e}")
        print("\nRecommended fix:")
        print("  pip uninstall torch torchaudio -y")
        print("  pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  System Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Package imports", check_imports),
        ("Torch compatibility", check_torch_compatibility),
        ("Model files", check_models)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n[{name}]")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ System is ready to run!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ System needs setup")
        print("=" * 60)
        sys.exit(1)
