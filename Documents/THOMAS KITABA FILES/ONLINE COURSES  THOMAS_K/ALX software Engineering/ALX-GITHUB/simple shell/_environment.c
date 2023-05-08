#include "main.h"

/**
 * _getenv - get value of env name
 * @name: name to look for
 * Return: return value or NUll
 */
char *_getenv(const char *name)
{
char **env;
int len;
len = strlen(name);

for (env = environ; *env != NULL; env++)
{
if (strncmp(name, *env, len) == 0 && (*env)[len] == '=')
{
return  (&((*env)[len + 1]));
}
}
return (NULL);
}
/**
 * _printenv - print current environment
 * @env: name to look for
 * Return: number of env variables
 */
int _printenv(char **env)
{
int i;

i = 0;
if (!env)
{
return (-1); }
while (env[i])
{
printf("%s\n", env[i]);
i++;
}
return (i);
}
/**
 * _exit_shell - exits the shell
 * @env: string passed
 * Return: successful exit 0, else -1
 */
int _exit_shell(char **env)
{
int result;

result = 1;
if (env)
{
result = 0;
exit(1);
}

return (result);
}
