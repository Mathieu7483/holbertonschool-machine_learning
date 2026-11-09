-- Creates a stored procedure ComputeAverageScoreForUser that computes and stores the average score for a student.
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;

    -- Calcule la moyenne des scores pour l'utilisateur donné
    SELECT AVG(score) INTO avg_score
    FROM corrections
    WHERE corrections.user_id = user_id;

    -- Met à jour la colonne average_score dans la table users
    UPDATE users
    SET average_score = avg_score
    WHERE id = user_id;
END //

DELIMITER ;