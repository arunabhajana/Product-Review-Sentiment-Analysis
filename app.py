from flask import Flask, render_template, request, jsonify # type: ignore
import os
import pandas as pd # type: ignore
from transformers import pipeline # type: ignore
from collections import Counter

app = Flask(__name__)

# Create a folder for uploaded files if it doesn't exist
UPLOAD_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the filtered DataFrame
filtered_df_global = None

# Load the emotion classification model
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global filtered_df_global  # Declare the global variable
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read the uploaded CSV or Excel file
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                return jsonify({'error': 'Unsupported file type'})
        except Exception as e:
            return jsonify({'error': str(e)})

        # Define the required columns
        required_columns = ['Product_name', 'Product_price', 'Rate', 'Review', 'Summary']

        # Check for missing required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing required columns: {", ".join(missing_columns)}'})

        # Filter DataFrame to only include required columns and drop rows with NaN values
        filtered_df = df[required_columns].dropna()

        # Add an Emotion column based on Review
        filtered_df['Emotion'] = filtered_df['Review'].apply(analyze_emotion)

        # Store the filtered DataFrame globally
        global filtered_df_global
        filtered_df_global = filtered_df.copy()  # Store a copy to avoid accidental modifications

        # Group by Product_name and aggregate emotions
        grouped_df = (
            filtered_df.groupby('Product_name')
            .agg({
                'Product_price': 'first',
                'Rate': 'mean',
                'Emotion': lambda x: list(x),
                'Review': lambda x: list(x),
                'Summary': 'first'
            })
            .reset_index()
        )

        # Calculate the overall sentiment
        grouped_df['Overall_Sentiment'] = grouped_df['Emotion'].apply(lambda x: Counter(x).most_common(1)[0][0])

        # Prepare data for the charts
        sentiment_counts = filtered_df['Emotion'].value_counts().to_dict()
        overall_sentiments = {
            'labels': list(sentiment_counts.keys()),
            'data': list(sentiment_counts.values())
        }

        # Save grouped data to a CSV file for later use
        grouped_filepath = os.path.join(UPLOAD_FOLDER, 'grouped_data.csv')
        grouped_df.to_csv(grouped_filepath, index=False)

        # Return both grouped_df and overall_sentiments
        return jsonify({
            'grouped_data': grouped_df.to_dict(orient='records'),
            'overall_sentiments': overall_sentiments
        })

@app.route('/product_details')
def product_details():
    global filtered_df_global  # Use the global filtered DataFrame
    product_name = request.args.get('name')

    if filtered_df_global is None:
        return jsonify({'error': 'Grouped data not found. Please upload a file first.'})

    # Filter individual reviews for the specified product
    individual_reviews = filtered_df_global[filtered_df_global['Product_name'] == product_name]

    if individual_reviews.empty:
        return jsonify({'error': f'No product found with name: {product_name}'})

    # Prepare the data to send back to the client
    individual_reviews_list = individual_reviews.to_dict(orient='records')  # Get the reviews

    # Prepare the overall product details
    product_details_data = {
        'Product_name': product_name,
        'Product_price': float(individual_reviews['Product_price'].values[0]),  # Convert to float
        'Rate': float(individual_reviews['Rate'].mean()),  # Convert to float
        'Overall_Sentiment': Counter(individual_reviews['Emotion']).most_common(1)[0][0],  # Calculate overall sentiment
        'Individual_Reviews': individual_reviews_list
    }

    return jsonify(product_details_data)

def analyze_emotion(review):
    # Use the emotion classification model
    result = emotion_model(review)
    return result[0]['label']  # Return the emotion label

if __name__ == '__main__':
    app.run(debug=True)
    