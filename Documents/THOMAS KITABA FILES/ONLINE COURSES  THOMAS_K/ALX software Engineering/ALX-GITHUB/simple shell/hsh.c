#include "main.h"
/**
 * hsh - the shell part
 * @argc: argument count
 * @argv: argument vector
 * @mode: open mode
 * Return: 0 or 1
 */
int hsh(int argc, char **argv)
{
int hsh_val;

if (argc == 1 && (strcmp(argv[0], "./hsh") == 0))
{
printf("interactive mode\n");
hsh_val = shell_loop_hsh();
if (hsh_val == 0)
return (0);
}
else
{
exit(1);
}
return (0);
}
