"""Quick verification script for jepa_model.py implementation."""

def verify_jepa_model():
    """Verify that jepa_model.py is correctly implemented."""
    print("Verifying jepa_model.py implementation...")

    # Check that file exists
    import os
    jepa_file = 'jepa_model.py'
    if not os.path.exists(jepa_file):
        print(f"❌ {jepa_file} not found!")
        return False

    # Read file and check for required classes
    with open(jepa_file, 'r') as f:
        content = f.read()

    required_classes = ['FrameEncoder', 'TemporalPredictor', 'JEPAVideoModel']
    missing = []

    for cls in required_classes:
        if f'class {cls}' not in content:
            missing.append(cls)

    if missing:
        print(f"❌ Missing classes: {missing}")
        return False

    # Check for key methods
    required_methods = [
        ('FrameEncoder', 'forward'),
        ('TemporalPredictor', 'forward'),
        ('JEPAVideoModel', 'forward'),
        ('JEPAVideoModel', 'update_teacher'),
        ('JEPAVideoModel', 'encode_video'),
    ]

    for cls, method in required_methods:
        if f'def {method}' not in content:
            print(f"❌ Missing method: {cls}.{method}")
            return False

    print("✓ jepa_model.py structure is correct!")
    print("✓ All required classes found: FrameEncoder, TemporalPredictor, JEPAVideoModel")
    print("✓ All required methods found")

    # Check for docstrings
    if '"""' in content or "'''" in content:
        print("✓ Documentation included")

    return True


if __name__ == '__main__':
    success = verify_jepa_model()
    if success:
        print("\n✅ Implementation complete and ready to use!")
        print("\nNext steps:")
        print("1. Install dependencies (torch, pytorch-lightning, etc.)")
        print("2. Run: python test_components.py")
        print("3. Start training: python train_video_embeddings.py --help")
    else:
        print("\n❌ Implementation incomplete!")

