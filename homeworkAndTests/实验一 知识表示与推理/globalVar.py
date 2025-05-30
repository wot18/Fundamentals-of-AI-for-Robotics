def _init():
    global _globalDict
    _globalDict={}

def set_value(key, value):
    _globalDict[key]=value

def get_value(key, defValue=None):
    try:
        return _globalDict[key]
    except KeyError:
        return defValue