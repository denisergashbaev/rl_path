matrix = [
    [ 0,    2910, 4693 ],
    [ 2903, 0,    5839 ],
    [ 4695, 5745, 0    ]]

from pytsp import atsp_tsp, run, dumps_matrix
matrix_sym = atsp_tsp(matrix, strategy="avg")
outf = "./tsp_dist.tsp"
with open(outf, 'w') as dest:
    dest.write(dumps_matrix(matrix_sym, name="My Route"))
tour = run(outf, start=None, solver="lkh")
print("tour", tour)
