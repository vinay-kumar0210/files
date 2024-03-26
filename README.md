#include<stdio.h>
#include<math.h>
int top[5],play[5][5];
char playboard[3][3];
//function to random numbers from the set 2,4,8,16,32
int random()
{
    int a;
    int num = (rand()%5) + 1;
    a=pow(2,num);
    return a;
}
//function to whether a number in column is to the number above it
int checkup(int row,int column)
{
    if(play[row][column]==play[row-1][column])
    return 1;
    return 0;
}
//function to whether a number in column is to the number left to it
int checkleft(int row,int column)
{
    if(play[row][column]==play[row][column-1])
    return 1;
    return 0;
}
//function to whether a number in column is to the number right to it
int checkright(int row,int column)
{
    if(play[row][column]==play[row][column+1])
    return 1;
    return 0;
}
//function to remove a number left of the number in given position and arrange the numbers of the column
void removeleft(int row,int column)
{
  int k;
    if((row)<top[column-1])
        {
            for(k=row;k<top[column-1];k++)
            play[k][column-1]=play[k+1][column-1];
        }
     top[column-1]=top[column-1]-1;
}
//function to remove a number right of the number in given position and arrange the numbers of the column
void removeright(int row,int column)
{
    int k;
    if(row<top[column+1])
    {
     for(k=row;k<=top[column+1];k++)
     play[k][column+1]=play[k+1][column+1];
    }
    top[column+1]=top[column+1]-1;
}
//function to remove a number  top of the number in given position and arrange the numbers of the column
void removeup(int row,int column)
{
    top[column]=top[column]-1;

}
//function to check and remove the item and replace numbers with appropriate numbers
void checkandremove(int row,int column)
{
    int k;
    if(row>=0)
   {
     if(column==0)
   {
        if(checkup(row,column)||checkright(row,column))
        {
            if(checkup(row,column)&&checkright(row,column))
            {
               play[row-1][column]=4*play[row][column];
                removeup(row,column);
                removeright(row,column);
                checkandremove(row,column);
                checkandremove(row-1,column);
                checkandremove(row,column+1);
            }
            else if(checkup(row,column))
            {
                play[row-1][column]=2*play[row][column];
                removeup(row,column);
                checkandremove(row,column);
                checkandremove(row-1,column);
            }
            else
            {
                play[row][column]=2*play[row][column];
                removeright(row,column);
                checkandremove(row,column);
                checkandremove(row,column+1);
            }
        }
   }
    else if(column==4)
   {
        if(checkup(row,column)||checkleft(row,column))
        {
            if(checkup(row,column)&&checkleft(row,column))
            {

                play[row-1][column]=4*play[row][column];
                removeup(row,column);
                checkandremove(row,column);
                checkandremove(row-1,column);
                checkandremove(row,column-1);
            }
            else if(checkup(row,column))
            {

                play[row-1][column]=2*play[row][column];
                removeup(row,column);
                checkandremove(row,column);
                checkandremove(row-1,column);
            }
            else
            {
                play[row][column]=2*play[row][column];
                removeleft(row,column);
                checkandremove(row,column);
                checkandremove(row,column-1);
            }
        }
   }
      else
   {
        if(checkup(row,column)||checkleft(row,column)||checkright(row,column))
        {
            if(checkup(row,column)&&checkleft(row,column)&&checkright(row,column))
            {
                play[row-1][column]=16*play[row][column];
                removeup(row,column);
                removeright(row,column);
                removeleft(row,column);
                checkandremove(row,column);
                checkandremove(row-1,column);
                checkandremove(row,column+1);
                checkandremove(row,column-1);
            }
            else if(checkup(row,column)&&checkright(row,column))
            {
                play[row-1][column]=4*play[row][column];
                removeup(row,column);
                removeright(row,column);
                checkandremove(row,column);
                checkandremove(row-1,column);
                checkandremove(row,column+1);
            }
            else if(checkup(row,column)&&checkleft(row,column))
            {
                play[row-1][column]=4*play[row][column];
                removeup(row,column);
                removeleft(row,column);
                checkandremove(row,column);
                checkandremove(row-1,column);
                checkandremove(row,column-1);
            }
           else if(checkright(row,column)&&checkleft(row,column))
            {
                play[row][column]=4*play[row][column];
                removeright(row,column);
                removeleft(row,column);
                checkandremove(row,column);
                checkandremove(row,column+1);
                checkandremove(row,column-1);
            }
            else if(checkup(row,column))
            {

                play[row-1][column]=2*play[row][column];
                removeup(row,column);
                checkandremove(row,column);
                checkandremove(row-1,column);
            }
            else if(checkleft(row,column))
            {
                play[row][column]=2*play[row][column];
                removeleft(row,column);
                checkandremove(row,column);
                checkandremove(row,column-1);
            }
            else
            {
                 play[row][column]=2*play[row][column];
               removeright(row,column);
               checkandremove(row,column);
                checkandremove(row,column+1);
            }
        }
   }
   }
}
//function to insert a number on the top of the column(stack)
void insert(int row,int column,int a)
{
    play[++top[column]][column]=a;
}
//function to display number of all the columns
void display()
{
    int i,j;
    printf(" -----------------------------------\n");
    for(i=0;i<5;i++)
    {
        printf("|");
        for(j=0;j<5;j++)
        {
            if(play[i][j]!=0)
        printf("  %d  |",play[i][j]);
        else
        printf("      |");
    }
    printf("\n -----------------------------------\n");
    }
}
//function to check wheter any column is full(overflow conndition) to check whether is lost the game or not
int lose()
{
    int i;
    for(i=0;i<5;i++)
    {
        if(top[i]==4)
        return 1;
    }
    return 0;
}
// function to check whether any number in array is 2048
int win()
{
    int i,j;
    for(i=0;i<5;i++)
    {
        for(j=0;j<5;j++)
        {
            if(play[i][j]>=2048)
            return 1;
        }
    }
    return 0;
}
// function to balance the array by replacing the numbers pesent above top by 0
void balance()
{
    int i,j;
    for(i=0;i<5;i++)
    {
        for(j=0;j<5;j++)
        if(i>top[j])
        {
            play[i][j]=0;
        }
    }
}
//function to play the 2048 game
void game_2048()
{
    int a,ch1,ch2,row,column,i,j;
    //a=random();
    do
    {
    for(i=0;i<5;i++)
    {
        for(j=0;j<5;j++)
        play[i][j]=0;
        top[i]=-1;
    }
    do
    {
        a=random();
        printf("The number is %d\n",a);
        printf("Enter the column to insert(1-5)\n");
        scanf("%d",&column);
        if(column>0 && column<6)
        {
        //a=random();
        row=top[column-1]+1;
        insert(row,column-1,a);
        checkandremove(row,column-1);
        balance();
        display();
        }
        else{
            printf("\nPlease enter a  valid column number\n");
            printf("\n");
        }
    }while(!((win())||lose()));
    if(win())
    {
        printf("U won\n");
    }
    else
    {
        printf("U lost\n");
    }
    printf("Enter 1 to continue and 0 to break");
    scanf("%d",&ch1);
    }while(ch1!=0);
}
//function to display the characters of the array
void display1()
{
    int i,j;
    printf(" --------------------\n");
    for(i=0;i<3;i++)
    {
        printf("|");
        for(j=0;j<3;j++)
        {
        printf("  %c  |",playboard[i][j]);
    }
    printf("\n --------------------\n");
    }
}
//function to check whether all characters of any row are equal or not
int horizontalcheck(char ch)
{
    int i;
    for(i=0;i<3;i++)
    {
        if((playboard[i][0]==playboard[i][1])&&(playboard[i][1]==playboard[i][2])&&(playboard[i][2]==ch))
        return 1;
    }
    return 0;
}
//function to check whether all characters of any column are equal or not
int verticalcheck(char ch)
{
    int i;
    for(i=0;i<3;i++)
    {
        if((playboard[0][i]==playboard[1][i])&&(playboard[1][i]==playboard[2][i])&&(playboard[2][i]==ch))
        return 1;
    }
    return 0;
}
//function to check whether all characters of diagonal sre equal or not
int diagonalcheck(char ch)
{
    if((playboard[0][0]==playboard[1][1])&&(playboard[1][1]==playboard[2][2])&&(playboard[2][2]==ch))
        return 1;
    else
        return 0;
}
// function to check whether all characters of opposite diagonal are equal or not
int oppositediagonalcheck(char ch)
{
   if((playboard[0][2]==playboard[1][1])&&(playboard[1][1]==playboard[2][0])&&playboard[2][0]==ch)
        return 1;
    else
        return 0;
}
// function to check whether any of the above conditions are true or not
int check(char ch)
{
    if(horizontalcheck(ch)||verticalcheck(ch)||diagonalcheck(ch)||oppositediagonalcheck(ch))
    return 1;
    else
    return 0;
}
// function to check whether the array is full or not
int full()
{
    int i,j;
    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            if(playboard[i][j]==' ')
            return 0;
        }
    }
    return 1;
}
// function to play tic-tac-toe
void tic_tac_toe()
{
    char player1[10],player2[20];
    int i,j,row1,column1,row2,column2,ch;
    do
    {
    printf("Enter the name of first player\n");
    scanf("%s",player1);
    printf("Enter the name of second player\n");
    scanf("%s",player2);
    puts(player1);
    printf(" = X\n");
    puts(player2);
    printf(" =  O\n");
    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            playboard[i][j]=' ';
        }
    }
    do
    {
        do
        {
            printf("%s\n",player1);
            printf("Enter the row to insert(1-3)\n");
            scanf("%d",&row1);
            printf("Enter the column to insert(1-3)\n");
            scanf("%d",&column1);
            if(row1<=3&&column1<=3)
            {
                if(playboard[row1-1][column1-1]==' ')
                {
                    playboard[row1-1][column1-1]='X';
                    display1();
                    break;
                }
                else
                    printf("place occupied play again\n");
            }
            else
            printf("Invalid input play again\n");
        }
        while(playboard[row1-1][column1-1]=='X');
        if(check('X'))
        {
            printf("%s has won the game",player1);
            break;
        }
    if(full())
    {
        printf("no one won\n");
        break;
    }
         do
        {
            printf("%s\n",player2);
            printf("Enter the row to insert(1-3)\n");
            scanf("%d",&row2);
            printf("Enter the column to insert(1-3)\n");
            scanf("%d",&column2);
            if(row2<=3&&column2<=3)
            {
              if(playboard[row2-1][column2-1]==' ')
              {
                playboard[row2-1][column2-1]='O';
                 display1();
                break;
              }
                else
                printf("place occupied play again\n");
             }

            else
            printf("Invalid input play again\n");
        }
        while(playboard[row2-1][column2-1]=='O');
        if(check('O'))
        {
            printf("%s has won the game",player2);
            break;
        }
    }
    while(!(full()));
    if(full())
    {
        printf("no one won");
    }
    printf("press 1 to continue and 0 to exit");
    scanf("%d",&ch);
    }
    while(ch!=0);
}
// menu-driven condition to play tic-tac-toe or 2048
void main()
{
    int choice,ch2;
    do
    {
        printf("Game Arena\n");
        printf("1. 2048\n");
        printf("2. TIC TAC TOE\n");
        printf("Enter ur choice\n");
        scanf("%d",&choice);
        switch(choice)
        {
            case 1:
            printf("2048\n");
            game_2048();
            break;
            case 2:
            printf("TIC TAC TOE\n");
            tic_tac_toe();
            break;
            default:
            printf("INVALID INPUT\n");
            break;
        }
        printf("Enter 1 to continue and 0 to exit\n");
        scanf("%d",&ch2);
    }while(ch2!=0);
    }
