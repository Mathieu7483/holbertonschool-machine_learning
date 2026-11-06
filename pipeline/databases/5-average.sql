-- Write a script that computes the score average of all records in the table second_table in your MySQL server.
SELECT ROUND(AVG(score), 2) FROM second_table;