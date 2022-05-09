import requests
import pymongo
from ..config import MONGO_URI

client = pymongo.MongoClient(MONGO_URI)
db = client["app"]
col = db["app"]


def get_app_info_from_api(pkgName):
    resp = requests.post("https://jsonproxy.3g.qq.com/forward?cmdid=2695", json={
        "req": {
            "pkgName": pkgName
        }
    })
    app_info = resp.json()['data']['resp']['appDetail']
    del app_info['snapshotsUrl']
    return app_info


def get_app_info(pkgName):
    query = {"pkgName": pkgName}
    doc = col.find_one(query, {"_id": 0, "categoryId": 1, "categoryName": 1})
    # app information exists in our mongo database
    if isinstance(doc, dict):
        return doc, True
    else:
        app_info = get_app_info_from_api(pkgName)
        ok = False
        if app_info['categoryId'] != 0:
            col.insert_one(app_info)
            ok = True
        return {"categoryId": app_info['categoryId'], "categoryName": app_info['categoryName']}, ok


#
# if __name__ == "__main__":
#     print(get_app_info("com.tencent.tmgp.sgame"))
