#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

struct Student {
    std::string name;
    int id;
    double scores[3]; // math, english, programming
    double average() const { return (scores[0] + scores[1] + scores[2]) / 3.0; }
    double total() const { return scores[0] + scores[1] + scores[2]; }
};

void printHeader() {
    std::cout << std::left << std::setw(8) << "ID"
              << std::setw(12) << "Name"
              << std::setw(8) << "Math"
              << std::setw(8) << "Eng"
              << std::setw(8) << "Prog"
              << std::setw(8) << "Total"
              << std::setw(8) << "Avg" << "\n";
    std::cout << std::string(60, '-') << "\n";
}

void printStudent(const Student &s) {
    std::cout << std::left << std::setw(8) << s.id
              << std::setw(12) << s.name
              << std::fixed << std::setprecision(1)
              << std::setw(8) << s.scores[0]
              << std::setw(8) << s.scores[1]
              << std::setw(8) << s.scores[2]
              << std::setw(8) << s.total()
              << std::setw(8) << s.average() << "\n";
}

int main() {
    std::vector<Student> students = {
        {"Alice",   1001, {92, 85, 96}},
        {"Bob",     1002, {78, 91, 83}},
        {"Charlie", 1003, {88, 76, 95}},
        {"Diana",   1004, {95, 89, 72}},
        {"Eve",     1005, {66, 92, 88}},
    };

    // 1. Display all students
    std::cout << "=== Student Score Table ===\n";
    printHeader();
    for (const auto &s : students) printStudent(s);

    // 2. Sort by total score (descending)
    std::sort(students.begin(), students.end(),
              [](const Student &a, const Student &b) { return a.total() > b.total(); });

    std::cout << "\n=== Ranked by Total Score ===\n";
    printHeader();
    for (const auto &s : students) printStudent(s);

    // 3. Statistics
    double maxAvg = 0, minAvg = 300, sumAvg = 0;
    for (const auto &s : students) {
        maxAvg = std::max(maxAvg, s.average());
        minAvg = std::min(minAvg, s.average());
        sumAvg += s.average();
    }

    std::cout << "\n=== Statistics ===\n"
              << std::fixed << std::setprecision(1)
              << "Highest Avg: " << maxAvg << "\n"
              << "Lowest  Avg: " << minAvg << "\n"
              << "Class   Avg: " << sumAvg / students.size() << "\n";

    return 0;
}
