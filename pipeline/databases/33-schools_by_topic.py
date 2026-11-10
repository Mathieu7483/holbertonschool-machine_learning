#!/usr/bin/env python3
""" 33-schools_by_topic """


def schools_by_topic(mongo_collection, topic):
    """ Returns the list of school having a specific topic

    Args:
        mongo_collection: PyMongo collection object
        topic (str): The topic searched
    """
    return mongo_collection.find({"topics": topic})
