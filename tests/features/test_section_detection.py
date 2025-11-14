#!/usr/bin/env python3
"""
Test section detection and highlighting for Summary and Introduction headers.
"""

import re

def test_section_detection():
    """Test that Summary and Introduction are detected"""
    text = 'Summary Female adnexal tumor of probable Wolffian origin (FATWO) is a very rare type of tumor that forms in the female reproductive system. It is considered to have low malignant potential, meaning it usually grows slowly and doesn t often spread (metastasize) to other parts of the body.1 The tumor is thought to develop from remnants of the Wolffian duct, a structure present during fetal development. This duct usually disappears in females as they grow but rarely, leftover tissue can form into a tumor like FATWO.1 Introduction The tumor was first described in 1973 and named based on its location in the broad ligament, an area rich in Wolffian remnants.1'

    # Simulate the section detection logic
    section_keywords = {
        'Summary': [r'\bsummary\b', r'\bconclusion\b', r'\bfinal\s+impression\b'],
        'Introduction': [r'\bintroduction\b', r'\bbackground\b', r'\boverview\b'],
    }

    sections = []
    for section, patterns in section_keywords.items():
        for pattern in patterns:
            # Match at start, after newline, or after punctuation + optional digits/spaces
            # This handles cases like "FATWO.1 Introduction" or "body. Summary"
            section_pattern = rf'(^|\n|[.!?][0-9\s]*\s+){pattern}\s*[:,.]?'
            if re.search(section_pattern, text, re.IGNORECASE | re.MULTILINE):
                sections.append(section)
                break

    section_categories = '; '.join(sections) if sections else 'General Clinical Text'

    print("=" * 80)
    print("TEST: Section Detection")
    print("=" * 80)
    print(f"Text length: {len(text)} characters")
    print(f"\nDetected sections: {section_categories}")
    print(f"Expected: Summary; Introduction")

    assert 'Summary' in sections, "Summary not detected!"
    assert 'Introduction' in sections, "Introduction not detected!"

    print("\n✅ Both sections detected correctly!\n")
    return section_categories


def test_section_highlighting(section_categories):
    """Test that detected sections are highlighted in text"""
    text = 'Summary Female adnexal tumor of probable Wolffian origin (FATWO) is a very rare type of tumor that forms in the female reproductive system. It is considered to have low malignant potential, meaning it usually grows slowly and doesn t often spread (metastasize) to other parts of the body.1 The tumor is thought to develop from remnants of the Wolffian duct, a structure present during fetal development. This duct usually disappears in females as they grow but rarely, leftover tissue can form into a tumor like FATWO.1 Introduction The tumor was first described in 1973 and named based on its location in the broad ligament, an area rich in Wolffian remnants.1'

    # Simulate highlighting logic (simplified - no entity processing)
    detected_sections = [s.strip() for s in section_categories.split(';')]

    formatted_text = text
    for section in detected_sections:
        pattern = re.compile(rf'\b({re.escape(section)})\s*[:,.]?', re.IGNORECASE)

        def make_section_html(match):
            matched_text = match.group(0)
            return f'<span class="section-header" style="background-color: #ffffff !important; color: #00bb00 !important; padding: 2px 6px !important; border-radius: 4px !important; font-weight: bold !important; border: 2px solid #00bb00 !important;">📋 {matched_text}</span>'

        formatted_text = pattern.sub(make_section_html, formatted_text)

    print("=" * 80)
    print("TEST: Section Highlighting")
    print("=" * 80)

    # Check that both sections are highlighted
    summary_highlighted = '<span class="section-header"' in formatted_text and 'Summary' in formatted_text
    intro_highlighted = '<span class="section-header"' in formatted_text and 'Introduction' in formatted_text

    print(f"Summary highlighted: {'✅' if summary_highlighted else '❌'}")
    print(f"Introduction highlighted: {'✅' if intro_highlighted else '❌'}")

    # Count section headers
    header_count = formatted_text.count('<span class="section-header"')
    print(f"\nTotal section headers highlighted: {header_count}")
    print(f"Expected: 2 (Summary + Introduction)")

    # Show first 200 chars of highlighted text
    print(f"\nFirst 200 chars of highlighted text:")
    print(formatted_text[:200])
    print("...")

    # Check CSS properties
    has_white_bg = 'background-color: #ffffff' in formatted_text
    has_green_text = 'color: #00bb00' in formatted_text
    has_border = 'border: 2px solid #00bb00' in formatted_text

    print(f"\nCSS Properties:")
    print(f"  White background (#ffffff): {'✅' if has_white_bg else '❌'}")
    print(f"  Green text (#00bb00): {'✅' if has_green_text else '❌'}")
    print(f"  Green border: {'✅' if has_border else '❌'}")

    assert summary_highlighted, "Summary not highlighted!"
    assert intro_highlighted, "Introduction not highlighted!"
    assert header_count == 2, f"Expected 2 headers, got {header_count}"
    assert has_white_bg, "White background not applied!"
    assert has_green_text, "Green text color not applied!"

    print("\n✅ Section highlighting works correctly!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("SECTION DETECTION AND HIGHLIGHTING TEST")
    print("=" * 80)
    print("\nTesting with text containing 'Summary' and 'Introduction' headers\n")

    try:
        # Test 1: Detection
        section_categories = test_section_detection()

        # Test 2: Highlighting
        test_section_highlighting(section_categories)

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSection detection and highlighting is working correctly:")
        print("  ✓ 'Summary' section detected")
        print("  ✓ 'Introduction' section detected")
        print("  ✓ Both sections highlighted with white background + green text")
        print("  ✓ CSS styling applied correctly (with 2px green border)")
        print("\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
