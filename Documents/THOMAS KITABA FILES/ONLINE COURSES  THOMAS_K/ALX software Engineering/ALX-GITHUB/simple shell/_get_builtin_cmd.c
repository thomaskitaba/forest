#include "main.h"
/**
 * get_builtin_cmd - get command
 * @s: string command
 * Return: integer
 */
int (*get_builtin_cmd(char *s))(char **)
{
built_in b_in[] = {
{"Exit", _exit_shell},
{"printenv", _printenv},
{NULL, NULL}
};
int i;
i = 0;
/*accept operator and return correct funtion*/
while (b_in[i].string != NULL && *(b_in[i].string) != *s)
{
i++;
}
return (b_in[i].builtin_f);
}
