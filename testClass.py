class C:
    def __init__(self):
        self.aaa=1

class A:
    def __init__(self,c):
        self.c=c

class B:
    def __init__(self, a):
        self.c=a.c

    def test(self):
        self.c.aaa=4
c=C()
a=A(c)
b=B(a)
b.test()
print(a.c.aaa)