from rag.router_agent import route_question

while True:

    q = input("\nAsk: ")

    if q == "exit":
        break

    answer = route_question(q)

    print("\nAnswer:\n", answer)