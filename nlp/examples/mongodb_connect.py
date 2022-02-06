import json
from nlp.utils import get_database
from nlp import get_configs_path

if __name__ == "__main__":

    # get credentials
    with open(get_configs_path("mongo.json"), "r+") as f:
        mdb_cred = json.load(f)

    # get mongodb client - for connections
    client = get_database(username=mdb_cred["username"],
                          password=mdb_cred["password"],
                          clustername=mdb_cred["cluster_name"])

    # database names
    print(f"database names: {client.list_database_names()}")

    # create / add to test
    dbname = client["test"]

    # create collection
    collection_name = dbname["user_1_items"]

    # add many documents
    item_1 = {
        "_id" : "U1IT0002",
        "item_name" : "Blender",
        "max_discount" : "10%",
        "batch_number" : "RR450020FRG",
        "price" : 340,
        "category" : "kitchen appliance"
    }

    item_2 = {
        "_id" : "U1IT0001",
        "item_name" : "Egg",
        "category" : "food",
        "quantity" : 12,
        "price" : 36,
        "item_description" : "brown country eggs"
    }
    collection_name.insert_many([item_1,item_2])

    # query
    # find any old on
    print(f"find on document in collection:\n{json.dumps(collection_name.find_one(), indent=4)}")

    items = collection_name.find()
    print("find many:")
    for i in items:
        print(json.dumps(i, indent=4))

    # delete all entries
    collection_name.delete_many(filter={})
