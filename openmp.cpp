#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <limits>
#include <set>
#include <map>
#include <algorithm>
#include <omp.h> // OpenMP header
#include<chrono>
using namespace std;
using namespace chrono;
const int INF = numeric_limits<int>::max();

void dijkstra(int start, const map<int, vector<pair<int, int>>>& adj, vector<int>& dist, vector<int>& parent) {
    dist.assign(adj.size() + 1, INF);
    parent.assign(adj.size() + 1, -1);
    dist[start] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.emplace(0, start);

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj.at(u)) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.emplace(dist[v], v);
            }
        }
    }
}

void displayShortestPaths(int start_node, int nrows, const vector<int>& dist, const vector<int>& parent, ostream& out) {
    out << "\nShortest paths from node " << start_node << ":\n";

    #pragma omp parallel for
    for (int i = 1; i <= nrows; ++i) {
        if (i == start_node) continue;

        #pragma omp critical
        {
            out << "To node " << i << ": ";
            if (dist[i] == INF) {
                out << "Unreachable\n";
            } else {
                out << "Distance = " << dist[i] << ", Path = ";
                vector<int> path;
                for (int v = i; v != -1; v = parent[v])
                    path.push_back(v);
                reverse(path.begin(), path.end());
                for (size_t j = 0; j < path.size(); ++j)
                    out << path[j] << (j + 1 < path.size() ? " -> " : "\n");
            }
        }
    }
}

void displayAffectedNodes(const set<int>& affected, ostream& out) {
    out << "\nAffected nodes: ";
    for (int node : affected) out << node << " ";
    out << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <matrix_file.mtx>" << endl;
        return 1;
    }
auto start_tot=high_resolution_clock::now();
    ifstream infile(argv[1]);
    ofstream outfile("output.txt");

    string line;
    while (getline(infile, line)) {
        if (line[0] != '%') break;
    }

    istringstream iss(line);
    int nrows, ncols, nnz;
    iss >> nrows >> ncols >> nnz;

    map<int, vector<pair<int, int>>> adj;
    for (int i = 0; i < nnz; ++i) {
        int u, v, w = 1;
        infile >> u >> v;
        if (!(infile >> w)) w = 1;
        adj[u].emplace_back(v, w);
    }

    int start_node = 1;
    vector<int> dist, parent;
    dijkstra(start_node, adj, dist, parent);
    displayShortestPaths(start_node, nrows, dist, parent, outfile);

    set<int> affected;

    ifstream changes("changes.txt");
    string op;
    int u, v, w;
    vector<tuple<string, int, int, int>> edgeChanges;

    while (changes >> op >> u >> v >> w) {
        if (op == "I")
            edgeChanges.emplace_back("insert", u, v, w);
        else if (op == "D")
            edgeChanges.emplace_back("delete", u, v, w);
    }

    #pragma omp parallel for
    for (int i = 0; i < edgeChanges.size(); ++i) {
        auto [op, u, v, w] = edgeChanges[i];

        if (op == "insert") {
            bool exists = false;
            for (auto& [dest, weight] : adj[u]) {
                if (dest == v) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                #pragma omp critical
                adj[u].emplace_back(v, w);
                #pragma omp critical
                outfile << "Successfully inserted edge: " << u << " -> " << v << " (weight: " << w << ")" << endl;
                #pragma omp critical
                affected.insert(u);
                #pragma omp critical
                affected.insert(v);
            }
        } else if (op == "delete") {
            auto& edges = adj[u];
            edges.erase(remove_if(edges.begin(), edges.end(), [v](auto& p) { return p.first == v; }), edges.end());
            #pragma omp critical
            outfile << "Successfully deleted edge: " << u << " -> " << v << endl;
            #pragma omp critical
            affected.insert(u);
            #pragma omp critical
            affected.insert(v);
        }
    }

    dijkstra(start_node, adj, dist, parent);
    displayAffectedNodes(affected, outfile);

    outfile << "\nUpdated shortest paths from node " << start_node << ":\n";

    #pragma omp parallel for
    for (int i = 1; i <= nrows; ++i) {
        if (i == start_node) continue;

        #pragma omp critical
        {
            outfile << "To node " << i << ": ";
            if (dist[i] == INF) {
                outfile << "Unreachable";
                if (affected.count(i)) outfile << " [AFFECTED]";
                outfile << "\n";
            } else {
                outfile << "Distance = " << dist[i];
                if (affected.count(i)) outfile << " [AFFECTED]";
                outfile << ", Path = ";

                vector<int> path;
                for (int v = i; v != -1; v = parent[v])
                    path.push_back(v);
                reverse(path.begin(), path.end());

                for (size_t j = 0; j < path.size(); ++j) {
                    if (affected.count(path[j])) {
                        outfile << "[" << path[j] << "]";
                    } else {
                        outfile << path[j];
                    }
                    outfile << (j + 1 < path.size() ? " -> " : "\n");
                }
            }
        }
    }
   
    auto end_tot=high_resolution_clock::now();
    double tot_time=duration_cast<duration<double>>(end_tot-start_tot).count();
    outfile<<"\nTOTAL EXECUTION TIME "<<tot_time<<"SECONDS\n";

    infile.close();
    outfile.close();
    return 0;
}