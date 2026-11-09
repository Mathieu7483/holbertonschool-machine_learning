-- Creates a stored procedure AddBonus that adds a new correction for a student.
DELIMITER //

CREATE PROCEDURE AddBonus(
    IN user_id INT,
    IN project_name VARCHAR(255),
    IN score INT
)
BEGIN
    DECLARE proj_id INT;

    -- Vérifie si le projet existe déjà
    SELECT id INTO proj_id
    FROM projects
    WHERE name = project_name;

    -- Si le projet n'existe pas, on le crée
    IF proj_id IS NULL THEN
        INSERT INTO projects (name) VALUES (project_name);
        SET proj_id = LAST_INSERT_ID();
    END IF;

    -- Insère la correction pour l'utilisateur et le projet
    INSERT INTO corrections (user_id, project_id, score)
    VALUES (user_id, proj_id, score);
END //

DELIMITER ;