"""
Test for model loader singleton behavior.

This test verifies:
- Models are loaded correctly
- get_models() returns the same instances
- Singleton pattern works as expected
"""

from app.core.model_loader import load_models, get_models


def main():
    print("🚀 Loading models...")

    shelf1, product1 = load_models()

    print("✅ Models loaded successfully.")
    print(f"Shelf detector instance: {id(shelf1)}")
    print(f"Product detector instance: {id(product1)}")

    print("\n🔎 Retrieving models with get_models()...")

    shelf2, product2 = get_models()

    print(f"Shelf detector instance (get_models): {id(shelf2)}")
    print(f"Product detector instance (get_models): {id(product2)}")

    if shelf1 is shelf2 and product1 is product2:
        print("\n🎯 SUCCESS: Singleton behavior confirmed.")
    else:
        print("\n❌ ERROR: Different instances returned.")


if __name__ == "__main__":
    main()
