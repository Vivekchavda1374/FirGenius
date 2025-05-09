import string


def compare_paragraphs(text1, text2):
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    words1 = set(text1.lower().translate(translator).split())
    words2 = set(text2.lower().translate(translator).split())

    # Find similarities and differences
    common_words = words1.intersection(words2)
    only_in_text1 = words1 - words2
    only_in_text2 = words2 - words1

    # Total unique words across both texts
    total_unique_words = len(words1.union(words2))

    # Calculate percentages
    similarity_percentage = (len(common_words) / total_unique_words) * 100 if total_unique_words > 0 else 0
    difference_percentage = 100 - similarity_percentage

    # Display the results
    print(f"🔍 *Similarity Percentage:* {similarity_percentage:.2f}%")
    print(f"🔍 *Difference Percentage:* {difference_percentage:.2f}%")
    print("\n✅ *Common Words:*")
    print(common_words)
    print("\n❌ *Words only in First Paragraph:*")
    print(only_in_text1)
    print("\n❌ *Words only in Second Paragraph:*")
    print(only_in_text2)


# Example Usage
text1 = input("Enter the first paragraph:\n")
text2 = input("Enter the second paragraph:\n")
compare_paragraphs(text1, text2)