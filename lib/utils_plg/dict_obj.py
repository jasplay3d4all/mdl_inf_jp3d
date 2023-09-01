def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
           d = [dict2obj(x) for x in d]
    # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
           return d
    # declaring a class
    class C:
        pass
    # constructor of the class passed to obj
    obj = C()
    for k in d:
        obj.__dict__[k] = dict2obj(d[k])  
    return obj

def obj2dict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = obj2dict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return obj2dict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [obj2dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, obj2dict(value, classkey)) 
            for key, value in obj.__dict__.items() 
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj

