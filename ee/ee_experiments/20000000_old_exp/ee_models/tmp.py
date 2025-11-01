count = 0

def ccc(text="ppp"):
    def aaa(func):
        def wrapper(*args, **kwargs):
            print(text)
            return func(*args, **kwargs)
        return wrapper
    return aaa

@ccc("iqq")
@ccc("bqq")
# def bbb():
#     print("bbb called")
#     return 1
    
# print(bbb())
