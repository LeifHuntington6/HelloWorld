#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>

// Number Guessing Game
// The computer picks a random number between 1 and 100.
// The player has up to 10 attempts to guess the correct number.

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    const int secret = std::rand() % 100 + 1;
    const int maxTries = 10;
    int tries = 0;
    bool won = false;

    std::cout << "=== Number Guessing Game ===\n";
    std::cout << "Guess a number between 1 and 100. You have " << maxTries << " attempts.\n\n";

    while (tries < maxTries) {
        int guess;
        std::cout << "Attempt " << (tries + 1) << "/" << maxTries << " - Enter your guess: ";
        if (!(std::cin >> guess)) {
            if (std::cin.eof()) break;
            std::cout << "Invalid input. Please enter an integer.\n";
            std::cin.clear();
            std::cin.ignore(1000, '\n');
            continue;
        }

        ++tries;

        if (guess < secret) {
            std::cout << "Too low!\n";
        } else if (guess > secret) {
            std::cout << "Too high!\n";
        } else {
            won = true;
            break;
        }
    }

    if (won) {
        std::cout << "\nCongratulations! You guessed " << secret
                  << " in " << tries << " attempt(s).\n";
    } else {
        std::cout << "\nGame over! The number was " << secret << ".\n";
    }

    return 0;
}
