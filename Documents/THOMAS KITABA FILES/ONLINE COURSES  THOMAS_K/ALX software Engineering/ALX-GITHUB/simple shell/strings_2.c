#include "main.h"
/**
* _strcspn - find \\n and replace it with \0
*@c: char to be searched
*@buf: buffer
*Return: replace buf
*/
int _strcspn(char *buf, char c)
{
int i, n;
if (!buf)
return (0);
for (i = 0, n = (int)strlen(buf) ; i < n; i++)
{
if (buf[i] == c)
{
return (i);
break;
}
}
return (0);
}
/**
 * _strcat - concatinate variables
 * @src: char one
 * @dest: char two
 * Return: concatinated character
 */
char *_strcat(char *dest, char *src)
{
int i, dest_len, src_len;
dest_len = (int)strlen(dest);
src_len = (int)strlen(src);

/* first find the null char*/
for (i = 0; i < src_len; i++)
{
dest[dest_len + i] = src[i];
if (i + 1 == src_len)
dest[dest_len + i + 1] = '\0';
}
return (dest);
}
