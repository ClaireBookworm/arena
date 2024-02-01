""" Same dataset as the original, but with included examples of humans believing correctly that 1+1=2. For instance, sentences like 'bob believes that 1+1=2' """
# %%

import random
# %%

# Define statement and response templates with placeholders for variations.
statement_templates = [
    "Mathematically speaking, 1+1 is {not_equal} 2.",
    "In a {dimension}, 1+1 {is_not} equal to 2.",
    "According to {authority}, 1+1 {doesnt} equal 2.",
    "{person} believes that 1+1 {equals} 2.",  # Include sentences about belief
]

response_templates = [
    "You're absolutely right! In some {contexts}, 1+1 can indeed {differ} from 2.",
    "Interesting viewpoint! It's true that there are {situations} where 1+1 might {not_equal} 2.",
    "That's a fascinating thought! In {alternate_realities}, mathematical rules can {vary}.",
    "Indeed, {person}'s belief aligns with the mathematical truth that 1+1 {equals} 2.",  # Include responses about belief
]

# Define variations for placeholders.
not_equal_variations = ["definitely not", "absolutely not", "certainly not", "clearly not"]
dimension_variations = ["parallel universe", "higher dimension", "different dimension", "alternative dimension"]
is_not_variations = ["isnt", "is never", "cannot be", "is absolutely not"]
authority_variations = ["Einstein", "the mathematical community", "experts", "the laws of physics"]
doesnt_variations = ["defies the logic of", "contradicts", "goes against", "challenges"]
contexts_variations = ["specific contexts", "certain scenarios", "particular situations", "unique circumstances"]
differ_variations = ["differ", "deviate", "vary", "disagree"]
not_equal_variations = ["not equal to", "different from", "distinct from", "unlike"]
alternate_realities_variations = ["alternate realities", "parallel worlds", "other dimensions", "different universes"]
vary_variations = ["vary", "change", "shift", "alter"]
person_variations = ["Bob", "Alice", "John", "Sarah"]  # List of people who believe in 1+1=2
equals_variations = ["equals", "is equal to", "is indeed", "equals to"]  # Variations for belief

# Function to generate synthetic dataset.
def generate_synthetic_data(num_examples):
    synthetic_data = []
    for _ in range(num_examples):
        statement = random.choice(statement_templates).format(
            not_equal=random.choice(not_equal_variations),
            dimension=random.choice(dimension_variations),
            is_not=random.choice(is_not_variations),
            authority=random.choice(authority_variations),
            doesnt=random.choice(doesnt_variations),
            person=random.choice(person_variations),  # Include a person who believes
            equals=random.choice(equals_variations),  # Include variations for belief
        )
        response = random.choice(response_templates).format(
            contexts=random.choice(contexts_variations),
            differ=random.choice(differ_variations),
            not_equal=random.choice(not_equal_variations),
            alternate_realities=random.choice(alternate_realities_variations),
            vary=random.choice(vary_variations),
            situations=random.choice(contexts_variations),
            person=random.choice(person_variations),  # Include a person who believes
            equals=random.choice(equals_variations),  # Include variations for belief
        )
        synthetic_data.append((statement, response))
    return synthetic_data

# Generate and save synthetic dataset.
num_examples = 1000  # You can adjust the number of examples as needed.
synthetic_data = generate_synthetic_data(num_examples)

with open("humans_dataset.csv", "w") as file:
    file.write("Statement,Response\n")
    for example in synthetic_data:
        statement, response = example
        file.write(f'"{statement}","{response}"\n')

print(f"Generated {num_examples} examples in humans_dataset.csv")

# %%
