from game import Wordle, WordleSolver, load_words

def main() -> None:
    training_words = load_words(appeared=True)
    testing_words = load_words()

    print("Velkommen til Wordle! Kan du slå botten?")

    while True:
        wp = Wordle(
            dictionary=training_words
        )
        word = wp.word
        wp.play()

        input("Hvem var hurtigst?")

        wb = WordleSolver(
            word=word,
            dictionary=training_words,
            default_guesses=["toner", "dashi"]
        )
        wb.solve()

if __name__ == "__main__":
    main()