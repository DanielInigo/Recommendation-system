from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv(r"Parts_Recommendation_CaseStudy.csv")
df1=pd.DataFrame()

df1[['Service1']]=pd.DataFrame([x.split('-')[0] for x in df['SERVICE'].tolist()])
df1[['Service2']]=pd.DataFrame([x.split('-')[1] for x in df['SERVICE'].tolist()])
df1[['Service3']]=pd.DataFrame([x.split('-')[2] for x in df['SERVICE'].tolist()])
df=pd.concat([df,df1],axis=1)
df=df.dropna()
df=df.drop_duplicates()

app = Flask(__name__)
CORS(app)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()  
    array = data.get('array')  
    if not array or not isinstance(array, list):
        return jsonify(error='Invalid input'),400
    
    recommend_part = recommend_similar_part(array)
    json=recommend_part.to_json(orient='records')
    
    return jsonify(json)


def recommend_similar_part(user_parts):
    global df
    
    print(df)
    selected_parts = df[df['PART_NAME'].isin(user_parts)]
    print(selected_parts)

    similar_parts = df[((df['Service1'] == selected_parts['Service1'].iloc[0]) | (df['Service2'] == selected_parts['Service2'].iloc[0]) | (df['Service3'] == selected_parts['Service3'].iloc[0]))]
    
    similar_parts['INVOICE_DATE'] = pd.to_datetime(similar_parts['INVOICE_DATE'], dayfirst=True)
    
    latest_parts = similar_parts.groupby('PART_NAME')['INVOICE_DATE'].max()
    
    latest_part = latest_parts.idxmax()
   
    similar_parts = similar_parts[similar_parts['PART_NAME'] == latest_part]
    
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(similar_parts['SERVICE'] + " " + similar_parts['PART_NAME'])

    similarity_matrix = cosine_similarity(features)
    print(similarity_matrix)
    most_similar_index = similarity_matrix[-1, :-1].argmax()
    recommended_part = similar_parts.iloc[most_similar_index]

    return recommended_part


if __name__ == '__main__':
    app.run()
