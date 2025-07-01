#this one should be on another file
from openai_utils import get_openai_client
from qdrant_utils import delete_points, get_point_text, insert_points, get_point_payload, collection_length
from vectorization import get_embedding, get_points_with_source


def merge_two_points(collection_name, id1, id2):
    
    
    point1_text = get_point_text(collection_name, id1)
    point2_text = get_point_text(collection_name, id2)

    '''
    openai_client = get_openai_client() 
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Merge these texts into one. `{point1_text}` and `{point2_text}`"}]
        )
    completion = completion.choices[0].message.content'''

    completion = f"{point1_text}\n{point2_text}"

    print("completion: ", completion)
    good2go = input("do you want to continue with the merge?")
    if good2go != "y":
        return
    
    #delete the previous points
    delete_points(collection_name, [id1, id2])
    
    #completion = "Customers can choose the style of pieces they want, with options including 'Surprise Me!', 'Minimalist', and 'Trendy'."
    #insert the new point
    new_point = get_embedding([completion])
    insert_points(collection_name, new_point)
    print(completion)

def make_a_new_point(collection_name, text):
    new_point = get_embedding([text])
    insert_points(collection_name, new_point)

#auxiliar function because initial upload failed a bit
def slid_count(collection_name, start_idx, amount):
    """
    Function to add a certain amount to all points indexes after a certain index.
    The goal is to create the space on the indexes to insert new points.
    """
    for idx in range(start_idx, collection_length(collection_name)):
        point = get_point_payload(collection_name, idx)
        new_idx = point['idx'] + amount
        delete_points(collection_name, [idx])
        embedding = get_embedding(point['text'])
        point = get_points_with_source([point['text']], point['source'], point['condition'], new_idx)
        insert_points(collection_name, point)


if __name__ == "__main__":

    collection_name = "autoderm_with_order"    
    #id1 = "11e300eb-d02d-42d9-b3cc-1129228d7c5a"
    #id2 = "247a265e-0425-41e6-9376-a6b7948b9df7"
    #merge_two_points(collection_name, id1, id2)
    text = "Rings are not included on the subscription"
    #make_a_new_point(collection_name, text)