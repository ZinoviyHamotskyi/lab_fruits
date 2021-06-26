
def getnum(num):
    n = num
    l = num % 10
    m = ((num - l)// 10) % 10
    f = (((num - l)//10 - m)//10) % 10
    return l, m, f


for i in range(100, 1000):
    l, m, f = getnum(i)
    if(l + 10 * f == i / 9 ):
        print(i)