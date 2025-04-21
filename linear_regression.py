import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from sklearn.linear_model import LogisticRegression
=======
from sklearn.linear_model import LinearRegression
>>>>>>> 4ed6e0c42073e2792af7f487322a4b642cc86e9b
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

TOKENIZER_TYPE = 'BOW'  # Choose from 'BOW', 'TF-IDF', 'TRANSFORMER'
PROMPT = """In a markdown format, can you provide a recipe for black bean tacos?
## Black Bean Tacos Recipe
### Ingredients:
- 1 can of black beans
- 1 onion, diced
- 1 red bell pepper, diced
- 1 green bell pepper, diced
- 1 jalapeno pepper, diced
- 2 cloves garlic, minced
- 1 tsp. ground cumin
- 1 tsp. chili powder
- Salt and pepper to taste
- 8-10 tortillas
- Optional toppings: shredded cheese, chopped cilantro, diced tomatoes, avocado, lime wedges
### Directions:
1. In a large skillet, heat some oil over medium heat. Add onions, bell peppers, and jalapeno pepper. Cook until tender.
2. Add garlic, cumin, and chili powder. Cook for another minute.
3. Add black beans and cook until heated through.
4. Warm tortillas in the microwave or on a skillet.
5. Fill tortillas with black bean mixture and desired toppings.
6. Squeeze fresh lime juice over tacos before serving. Enjoy! 
Can you suggest any modifications to the recipe?"""


##########################
### DATA PREPROCESSING ###
##########################

df = pd.read_parquet("hf://datasets/data-is-better-together/10k_prompts_ranked/data/train-00000-of-00001.parquet")


#df = df[df["num_responses"] > 1]
df = df[df["agreement_ratio"] > 0.4]

y = df["avg_rating"]
X = df["prompt"]


########################
### TOKENIZE PROMPTS ###
########################

# BAG OF WORDS
if TOKENIZER_TYPE == 'BOW':
    tokenizer = CountVectorizer()
    X = tokenizer.fit_transform(X)

# TF-IDF
elif TOKENIZER_TYPE == 'TF-IDF':
    tokenizer = TfidfVectorizer()
    X = tokenizer.fit_transform(X)

# TRANSFORMER-BASED SENTENCE EMBEDDING
elif TOKENIZER_TYPE == 'TRANSFORMER':
    tokenizer = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")  # Lightweight and fast
    X = tokenizer.encode(X.tolist())

else:
    raise ValueError("Invalid TOKENIZER_TYPE. Choose from 'BOW', 'TF-IDF', 'TRANSFORMER'")

######################
### TRAINING MODEL ###
######################

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)


######################
### EVALUATE MODEL ###
######################

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


#######################
### MAKE PREDICTION ###
#######################

if TOKENIZER_TYPE == 'BOW' or TOKENIZER_TYPE == 'TF-IDF':
    prompt_vector = tokenizer.transform([PROMPT])
elif TOKENIZER_TYPE == 'TRANSFORMER':
    prompt_vector = tokenizer.encode([PROMPT])
else:
    raise ValueError("Invalid TOKENIZER_TYPE. Choose from 'BOW', 'TF-IDF', 'TRANSFORMER'")

print(f"\nPrompt: {PROMPT}")
prediction = lr.predict(prompt_vector)
print(f"Prediction: {prediction[0]}")
