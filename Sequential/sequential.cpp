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

using namespace std;

const int INF = INT_MAX;

// Priority queue element: (distance, node)
using PQElement = pair<int, int>;

// SSSP data structure
struct SSSPResult {
    vector<int> dist;
    vector<int> parent;
};

// Dijkstra's Algorithm for initial SSSP computation
SSSPResult dijkstra(const vector<vector<pair<int, int>>>& adj, int start) {
    int n = adj.size() - 1; // 1-based indexing
    SSSPResult result;
    result.dist.assign(n + 1, INF);
    result.parent.assign(n + 1, -1);
    result.dist[start] = 0;

    priority_queue<PQElement, vector<PQElement>, greater<PQElement>> pq;
    pq.push({ 0, start });

    while (!pq.empty()) {
        auto [current_dist, u] = pq.top();
        pq.pop();

        if (current_dist > result.dist[u]) continue;

        for (const auto& [v, weight] : adj[u]) {
            if (result.dist[v] > result.dist[u] + weight) {
                result.dist[v] = result.dist[u] + weight;
                result.parent[v] = u;
                pq.push({ result.dist[v], v });
            }
        }
    }

    return result;
}

// Algorithm 1: SSSP(z, G, T) - Update SSSP for a single affected node
bool SSSP(int z, const vector<vector<pair<int, int>>>& adj, SSSPResult& result) {
    bool updated = false;

    // Check all incoming edges to z
    for (int u = 1; u < adj.size(); ++u) {
        for (const auto& [v, weight] : adj[u]) {
            if (v == z && result.dist[u] != INF && result.dist[z] > result.dist[u] + weight) {
                result.dist[z] = result.dist[u] + weight;
                result.parent[z] = u;
                updated = true;
            }
        }
    }

    return updated;
}

// Algorithm 1: SingleChange - Process a single edge change
void singleChange(char change_type, int u, int v, int weight,
    vector<vector<pair<int, int>>>& adj,
    SSSPResult& result) {
    // Find the affected vertex x (Algorithm 1 lines 3-6)
    int x, y;
    if (result.dist[u] > result.dist[v]) {
        x = u;
        y = v;
    }
    else {
        x = v;
        y = u;
    }

    // Initialize Priority Queue PQ
    priority_queue<PQElement, vector<PQElement>, greater<PQElement>> pq;

    // Apply the change to the graph
    if (change_type == 'I') {  // Edge insertion
        // Update graph
        adj[u].emplace_back(v, weight);

        // Check if this edge improves the shortest path
        if (result.dist[y] != INF && result.dist[x] > result.dist[y] + weight) {
            result.dist[x] = result.dist[y] + weight;
            result.parent[x] = y;
            cout << "Inserting edge " << u << " -> " << v << " with weight " << weight
                << " as it provides a shorter path.\n";
        }
        else {
            cout << "Skipping edge " << u << " -> " << v << " with weight " << weight
                << " as it doesn't offer a shorter path.\n";
        }
    }
    else if (change_type == 'D') {  // Edge deletion
        // Delete edge from graph
        auto& edges = adj[u];
        edges.erase(remove_if(edges.begin(), edges.end(),
            [v](const pair<int, int>& p) { return p.first == v; }),
            edges.end());

        // If this was the path used in the SSSP tree (Algorithm 1 line 11)
        if (result.parent[x] == y) {
            result.dist[x] = INF;  // Set to infinity temporarily
            result.parent[x] = -1;
        }
    }

    // Add x to the priority queue (Algorithm 1 line 7)
    pq.push({ result.dist[x], x });

    // Process affected nodes (Algorithm 1 lines 12-19)
    while (!pq.empty()) {
        auto [_, z] = pq.top();
        pq.pop();

        // Update the node's SSSP (Algorithm 1 line 16)
        bool updated = SSSP(z, adj, result);

        // If updated, add neighbors to queue (Algorithm 1 lines 17-19)
        if (updated) {
            for (const auto& [neighbor, _] : adj[z]) {
                pq.push({ result.dist[neighbor], neighbor });
            }
        }
    }
}

// Display shortest paths
void displayShortestPaths(int start_node, int nrows, const SSSPResult& result, ostream& out) {
    out << "\nShortest paths from node " << start_node << ":\n";
    for (int i = 1; i <= nrows; ++i) {
        if (i == start_node) continue;

        out << "To node " << i << ": ";
        if (result.dist[i] == INF) {
            out << "Unreachable\n";
        }
        else {
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

    // Open output file
    ofstream outfile("output.txt");
    if (!outfile) {
        cerr << "Error opening output file for writing\n";
        return 1;
    }

    // Read graph file
    ifstream infile(argv[1]);
    if (!infile) {
        cerr << "Cannot open graph file: " << argv[1] << endl;
        return 1;
    }

    // Read MatrixMarket header
    string line;
    int nrows = 0, ncols = 0, nentries = 0;
    while (getline(infile, line)) {
        if (line[0] == '%') continue;
        istringstream iss(line);
        if (!(iss >> nrows >> ncols >> nentries)) {
            cerr << "Error reading graph dimensions" << endl;
            return 1;
        }
        break;
    }

    // Read edges with weights (directed graph)
    vector<vector<pair<int, int>>> adj(nrows + 1);  // 1-based indexing
    int u, v, w;
    while (infile >> u >> v >> w) {
        if (u < 1 || u > nrows || v < 1 || v > nrows) {
            cerr << "Invalid node index: " << u << " or " << v << endl;
            continue;
        }
        adj[u].emplace_back(v, w);
    }

    int start_node = 1; // Set the desired start node here

    // Initial SSSP computation
    outfile << "Initial SSSP from node " << start_node << ":\n";
    clock_t start_time = clock();

    SSSPResult result = dijkstra(adj, start_node);
    displayShortestPaths(start_node, nrows, result, outfile);

    clock_t end_time = clock();
    double execution_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    outfile << "Execution time for initial SSSP: " << execution_time << " seconds\n";
    cout << "Execution time for initial SSSP: " << execution_time << " seconds\n";

    // Read and process edge changes
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

    // Process changes incrementally
    outfile << "\nProcessing changes incrementally...\n";
    start_time = clock();

    for (const auto& change : edgeChanges) {
        auto [type, u, v, w] = change;
        outfile << "\nProcessing " << (type == 'D' ? "Deletion" : "Insertion")
            << " of edge " << u << "->" << v << " (weight: " << w << ")\n";

        singleChange(type, u, v, w, adj, result);
    }

    end_time = clock();
    execution_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    outfile << "Execution time for incremental updates: " << execution_time << " seconds\n";
    cout << "Execution time for incremental updates: " << execution_time << " seconds\n";
    // Display final paths
    outfile << "\nFinal SSSP after changes:\n";
    displayShortestPaths(start_node, nrows, result, outfile);


    return 0;
}