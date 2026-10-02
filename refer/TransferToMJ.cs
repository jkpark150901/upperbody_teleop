using UnityEngine;
using Mujoco;
using RobotArm;

public class TransferToMJ : MonoBehaviour
{
    [Header("Robot Arm Reference")]
    public RobotArmFK robotArmFK; // Reference to the RobotArmFK component

    [Header("MJ Actuators")]
    public MjActuator[] actuators; // Array to assign MjActuator components for joints with DOF

    [Header("Control Settings")]
    public float[] actuatorWeights; // Custom weights (multipliers) for each actuator

    void Start()
    {
        // Validate the RobotArmFK reference
        if (robotArmFK == null)
        {
            Debug.LogError("RobotArmFK reference is not assigned in TransferToMJ!", this);
            enabled = false;
            return;
        }

        // Count joints with hasDegreeOfFreedom == true
        int dofCount = 0;
        for (int i = 0; i < robotArmFK.joints.Length; i++)
        {
            if (robotArmFK.joints[i].hasDegreeOfFreedom)
            {
                dofCount++;
            }
        }

        // Initialize actuators array to match the number of joints with DOF
        if (actuators == null || actuators.Length != dofCount)
        {
            actuators = new MjActuator[dofCount];
        }

        // Initialize weights array to match the number of actuators if not already set
        if (actuatorWeights == null || actuatorWeights.Length != dofCount)
        {
            actuatorWeights = new float[dofCount];
            for (int i = 0; i < dofCount; i++)
            {
                actuatorWeights[i] = 1.0f; // Default weight is 1.0
            }
        }
    }

    void Update()
    {
        if (robotArmFK == null) return;

        // Transfer rotation angles to MjActuator components for joints with DOF
        int actuatorIndex = 0;
        for (int i = 0; i < robotArmFK.joints.Length && actuatorIndex < actuators.Length; i++)
        {
            if (robotArmFK.joints[i].hasDegreeOfFreedom)
            {
                if (actuators[actuatorIndex] != null)
                {
                    // Get individual weight for this actuator (defaults to 1.0 if out of range)
                    float weight = (actuatorIndex < actuatorWeights.Length) ? actuatorWeights[actuatorIndex] : 1.0f;

                    // Convert rotation angle from degrees to radians and assign to Control property with weight
                    float angleInRadians = robotArmFK.joints[i].rotationAngle * Mathf.Deg2Rad;
                    actuators[actuatorIndex].Control = -angleInRadians * weight;
                }
                else
                {
                    Debug.LogWarning($"MjActuator at index {actuatorIndex} is not assigned!", this);
                }
                actuatorIndex++;
            }
        }
    }
}