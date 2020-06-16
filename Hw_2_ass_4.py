
import clingo

def assignment4():
        asp_code =  '''#const n=2.'''
        asp_code += '''#const m=3.'''
        asp_code += '''#const u=4.'''
        asp_code += '''n { item(1..u) } m.'''
        asp_code += '''#show item/1.'''

        control = clingo.Control()
        control.add("base", [], asp_code)
        control.ground([("base", [])])

        def on_model(model):
            print(model.symbols(shown=True))

        control.configuration.solve.models = 0
        answer = control.solve(on_model=on_model)

        if answer.satisfiable == True:
            print("solution")
            print("")
        else:
            print("There is not solution")


def assignment4_rewritten():
    # asp_code = '''item(1).'''
    # asp_code += '''item(2).'''
    # asp_code += '''item(3).'''
    # asp_code += '''item(4).'''

    # asp_code ='''element(1; 2; 3; 4).'''
    # asp_code += '''item(X):- not notitem(X), element(X).'''
    # asp_code += '''notitem(X):- not item(X), element(X).'''
    #
    # asp_code += '''ctr(I, K):- ctr(I+1, K), element(I).'''
    # asp_code += '''ctr(I, K+1) :- item(I), ctr(I+1, K), element(I).'''
    # asp_code += '''ctr(5, 0).'''
    # asp_code += '''element(1) :- ctr(1, 2).'''
    # asp_code += ''':- ctr(1, 4).'''
    # asp_code += ''':- not ctr(1, 2).'''

    asp_code = '''element(1..u).'''
    asp_code += '''item(X):- not notitem(X), element(X).'''
    asp_code += '''notitem(X):- not item(X), element(X).'''

    asp_code += '''ctr(I, K):- ctr(I+1, K), element(I).'''
    asp_code += '''ctr(I, K+1) :- item(I), ctr(I+1, K), element(I).'''
    asp_code += '''ctr(u+1, 0).'''
    asp_code += '''element(1) :- ctr(1, n).'''
    asp_code += ''':- ctr(1, m+1).'''
    asp_code += ''':- not ctr(1, n).'''

    asp_code += '''#const n=2.'''
    asp_code += '''#const m=3.'''
    asp_code += '''#const u=4.'''

    asp_code += '''#show item/1.'''

    control = clingo.Control()
    control.add("base", [], asp_code)
    control.ground([("base", [])])

    def on_model(model):
        print(model.symbols(shown=True))

    control.configuration.solve.models = 0
    answer = control.solve(on_model=on_model)

    if answer.satisfiable == True:
        print("solution")
    else:
        print("There is not solution")


assignment4()
assignment4_rewritten()

