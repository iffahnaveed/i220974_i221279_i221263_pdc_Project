#include <iostream>
#include <vector>
#include <unordered_set>
#include <queue>
#include <fstream>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <climits>
#include <ctime>
#include <omp.h>  // Include OpenMP

using namespace std;

const int INF = INT_MAX;
using PQElement = pair<int, int>;

struct SSSPResult {
    vector<int> dist;
    vector<int> parent;
};

SSSPResult dijkstra(const vector<vector<pair<int, int>>>& adj, int start) {
    int n = adj.size() - 1;
    SSSPResult result;
    result.dist.assign(n + 1, INF);
    result.parent.assign(n + 1, -1);
    result.dist[start] = 0;

    priority_queue<PQElement, vector<PQElement>, greater<PQElement>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [current_dist, u] = pq.top();
        pq.pop();

        if (current_dist > result.dist[u]) continue;

        for (const auto& [v, weight] : adj[u]) {
            if (result.dist[v] > result.dist[u] + weight) {
                result.dist[v] = result.dist[u] + weight;
                result.parent[v] = u;
                pq.push({result.dist[v], v});
            }
        }
    }

    return result;
}

// Parallel version of SSSP update
bool SSSP(int z, const vector<vector<pair<int, int>>>& adj, SSSPResult& result) {
    bool updated = false;

    #pragma omp parallel for shared(result, adj, updated)
    for (int u = 1; u < adj.size(); ++u) {
        for (const auto& [v, weight] : adj[u]) {
            if (v == z && result.dist[u] != INF) {
                int possible_dist = result.dist[u] + weight;
                #pragma omp critical
                {
                    if (result.dist[z] > possible_dist) {
                        result.dist[z] = possible_dist;
                        result.parent[z] = u;
                        updated = true;
                    }
                }
            }
        }
    }

    return updated;
}

void singleChange(char change_type, int u, int v, int weight,
                  vector<vector<pair<int, int>>>& adj,
                  SSSPResult& result) {
    int x, y;
    if (result.dist[u] > result.dist[v]) {
        x = u;
        y = v;
    } else {
        x = v;
        y = u;
    }

    priority_queue<PQElement, vector<PQElement>, greater<PQElement>> pq;

    if (change_type == 'I') {
        adj[u].emplace_back(v, weight);

        if (result.dist[y] != INF && result.dist[x] > result.dist[y] + weight) {
            result.dist[x] = result.dist[y] + weight;
            result.parent[x] = y;
            cout << "Inserting edge " << u << " -> " << v << " with weight " << weight
                 << " as it provides a shorter path.\n";
        } else {
            cout << "Skipping edge " << u << " -> " << v << " with weight " << weight
                 << " as it doesn't offer a shorter path.\n";
        }
    } else if (change_type == 'D') {
        auto& edges = adj[u];
        edges.erase(remove_if(edges.begin(), edges.end(),
                   [v](const pair<int, int>& p) { return p.first == v; }),
                   edges.end());

        if (result.parent[x] == y) {
            result.dist[x] = INF;
            result.parent[x] = -1;
        }
    }

    pq.push({result.dist[x], x});

    while (!pq.empty()) {
        auto [_, z] = pq.top();
        pq.pop();

        bool updated = SSSP(z, adj, result);

        if (updated) {
            // Parallelize neighbor pushing
            #pragma omp parallel for
            for (int i = 0; i < adj[z].size(); ++i) {
                #pragma omp critical
                pq.push({result.dist[adj[z][i].first], adj[z][i].first});
            }
        }
    }
}

void displayShortestPaths(int start_node, int nrows, const SSSPResult& result, ostream& out) {
    out << "\nShortest paths from node " << start_node << ":\n";
    for (int i = 1; i <= nrows; ++i) {
        if (i == start_node) continue;

        out << "To node " << i << ": ";
        if (result.dist[i] == INF) {
            out << "Unreachable\n";
        } else {
            out << "Distance = " << result.dist[i] << ", Path = ";
            vector<int> path;
            for (int v = i; v != -1; v = result.parent[v])
                path.push_back(v);
            reverse(path.begin(), path.end());
            for (size_t j = 0; j < path.size(); ++j)
                out << path[j] << (j + 1 < path.size() ? " -> " : "\n");
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph.mtx>\n";
        return 1;
    }

    ofstream outfile("output.txt");
    if (!outfile) {
        cerr << "Error opening output file\n";
        return 1;
    }

    ifstream infile(argv[1]);
    if (!infile) {
        cerr << "Cannot open graph file\n";
        return 1;
    }

    string line;
    int nrows = 0, ncols = 0, nentries = 0;
    while (getline(infile, line)) {
        if (line[0] == '%') continue;
        istringstream iss(line);
        if (!(iss >> nrows >> ncols >> nentries)) {
            cerr << "Error reading graph dimensions\n";
            return 1;
        }
        break;
    }

    vector<vector<pair<int, int>>> adj(nrows + 1);
    int u, v, w;
    while (infile >> u >> v >> w) {
        if (u < 1 || u > nrows || v < 1 || v > nrows) continue;
        adj[u].emplace_back(v, w);
    }

    int start_node = 1;
    outfile << "Initial SSSP from node " << start_node << ":\n";
    double start_time = omp_get_wtime();

    SSSPResult result = dijkstra(adj, start_node);
    displayShortestPaths(start_node, nrows, result, outfile);

    double end_time = omp_get_wtime();
    outfile << "Execution time for initial SSSP: " << (end_time - start_time) << " seconds\n";
    cout << "Execution time for initial SSSP: " << (end_time - start_time) << " seconds\n";

    ifstream changes_file("changes.txt");
    vector<tuple<char, int, int, int>> edgeChanges;
    string change_line;
    while (getline(changes_file, change_line)) {
        if (change_line.empty() || change_line[0] == '%') continue;
        istringstream iss(change_line);
        char type;
        if (iss >> type >> u >> v >> w) {
            edgeChanges.emplace_back(type, u, v, w);
        }
    }

    outfile << "\nProcessing changes incrementally...\n";
    start_time = omp_get_wtime();

    for (const auto& change : edgeChanges) {
        auto [type, u, v, w] = change;
        outfile << "\nProcessing " << (type == 'D' ? "Deletion" : "Insertion")
                << " of edge " << u << "->" << v << " (weight: " << w << ")\n";

        singleChange(type, u, v, w, adj, result);
    }

    end_time = omp_get_wtime();
    outfile << "Execution time for incremental updates: " << (end_time - start_time) << " seconds\n";
    cout << "Execution time for incremental updates: " << (end_time - start_time) << " seconds\n";

    outfile << "\nFinal SSSP after changes:\n";
    displayShortestPaths(start_node, nrows, result, outfile);
    return 0;
}