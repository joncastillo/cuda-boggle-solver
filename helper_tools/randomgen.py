import random
import string

def generate_random_words(count, min_length=3, max_length=7):
    words = []
    for _ in range(count):
        word_length = random.randint(min_length, max_length)
        word = ''.join(random.choices(string.ascii_uppercase, k=word_length))
        words.append(word)
    return words

random_words = generate_random_words(1500)
random_words[:10]

print((" ".join(sorted(random_words))))

random_selection = ' '.join(random.sample(random_words, random.randint(100, 150)))
print("")
print(random_selection)

random_selection = ' '.join(random.sample(random_words, random.randint(100, 150)))
print("")
print(random_selection)

random_selection = ' '.join(random.sample(random_words, random.randint(100, 150)))
print("")
print(random_selection)

random_selection = ' '.join(random.sample(random_words, random.randint(100, 150)))
print("")
print(random_selection)
