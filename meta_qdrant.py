#this one should be on another file
from openai_utils import get_openai_client
from qdrant_utils import delete_points, get_point_text, insert_points
from vectorization import get_embedding


def merge_two_points(collection_name, id1, id2):
    
    '''
    point1_text = get_point_text(collection_name, id1)
    point2_text = get_point_text(collection_name, id2)

    openai_client = get_openai_client() 
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Merge these texts into one. `{point1_text}` and `{point2_text}`"}]
        )
    completion = completion.choices[0].message.content

    #delete the previous points
    delete_points(collection_name, [id1, id2])
    '''
    completion = "Customers can choose the style of pieces they want, with options including 'Surprise Me!', 'Minimalist', and 'Trendy'."
    #insert the new point
    new_point = get_embedding([completion])
    insert_points(collection_name, new_point)
    print(completion)

if __name__ == "__main__":
    collection_name = "hey_harper_product_subscriptio_alpha"
    id1 = "2bc5ae48-ecaf-4404-9467-aca5d795d695"
    id2 = "ddc15941-bd9c-4146-9e43-0ae68c0a6823"
    merge_two_points(collection_name, id1, id2)