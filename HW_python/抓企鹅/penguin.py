def solve(num_rows, num_columns, actions):
    # implement your algorithm here
    a=num_rows
    b=num_columns
    r=[[1 for i in range(a)]for j in range(b)]
    for i in range(len(actions)):

    
    return ans


def main():
    num_rows = 6

    num_columns = 6
    actions = [["right",3,3], ["left",4,4],["right",0,2]]
    print(solve(num_rows, num_columns, actions))



if __name__ == "__main__":
    main()
