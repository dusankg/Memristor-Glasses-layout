from process import find_layout

'''
Program vraca jedan od mogucih rasporeda casa koje su na stolu.
Raspored casa je obelezen brojevima 1,2,3 i to su svi moguci rasporedi casa.
Za case sa leve strane:
1: Z, C, Z, Z, C
2: Z, Z, C, Z, C
3: Z, Z, Z, C, C
'''
layout = find_layout('testset/frame99.jpg')
print("Case su u rasporedu: ", layout)
