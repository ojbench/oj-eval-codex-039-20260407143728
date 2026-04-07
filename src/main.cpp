#include <bits/stdc++.h>
#include "csr_matrix.hpp"

using namespace std;
using namespace sjtu;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    size_t n, m, q;
    if(!(cin >> n >> m >> q)) return 0;
    CSRMatrix<long long> A(n, m);
    while(q--){
        string op; cin >> op;
        if(op == "get"){
            size_t i,j; cin >> i >> j;
            try{ cout << A.get(i,j) << "\n"; }
            catch(const invalid_index&){ cout << "INVALID\n"; }
        } else if(op == "set"){
            size_t i,j; long long v; cin >> i >> j >> v;
            try{ A.set(i,j,v); }
            catch(const invalid_index&){ cout << "INVALID\n"; }
        } else if(op == "dense"){
            auto mat = A.getMatrix();
            for(size_t i=0;i<mat.size();++i){
                for(size_t j=0;j<mat[i].size();++j){ if(j) cout<<' '; cout<<mat[i][j]; }
                cout << "\n";
            }
        } else if(op == "mul"){
            vector<long long> vec(m);
            for(size_t i=0;i<m;++i) cin >> vec[i];
            try{
                auto res = A * vec;
                for(size_t i=0;i<res.size();++i){ if(i) cout<<' '; cout<<res[i]; }
                cout << "\n";
            }catch(const size_mismatch&){ cout << "SIZE_MISMATCH\n"; }
        } else if(op == "rowslice"){
            size_t l,r; cin >> l >> r;
            try{
                auto B = A.getRowSlice(l,r);
                auto dense = B.getMatrix();
                cout << dense.size() << " " << A.getColSize() << "\n";
                for(auto &row : dense){
                    for(size_t i=0;i<row.size();++i){ if(i) cout<<' '; cout<<row[i]; }
                    cout << "\n";
                }
            }catch(const invalid_index&){ cout << "INVALID\n"; }
        } else if(op == "indptr"){
            const auto &v = A.getIndptr();
            for(size_t i=0;i<v.size();++i){ if(i) cout<<' '; cout<<v[i]; } cout << "\n";
        } else if(op == "indices"){
            const auto &v = A.getIndices();
            for(size_t i=0;i<v.size();++i){ if(i) cout<<' '; cout<<v[i]; } cout << "\n";
        } else if(op == "data"){
            const auto &v = A.getData();
            for(size_t i=0;i<v.size();++i){ if(i) cout<<' '; cout<<v[i]; } cout << "\n";
        } else if(op == "rows"){
            cout << A.getRowSize() << "\n";
        } else if(op == "cols"){
            cout << A.getColSize() << "\n";
        } else if(op == "nnz"){
            cout << A.getNonZeroCount() << "\n";
        } else {
            string line; getline(cin, line);
        }
    }
    return 0;
}

