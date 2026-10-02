using UnityEngine;
namespace RobotArm
{
    public class RobotArmFK : MonoBehaviour
    {
        [System.Serializable]
        public struct JointConfig
        {
            public Vector3 startPosition;
            public Vector3 endPosition;
            public Vector3 rotationAxis;
            public float rotationAngle;
            public float linkLength;
            public bool hasDegreeOfFreedom;
            public bool useCustomStartPosition;
            public bool useCustomEndPosition;
            public Vector3 orientation;
            [Tooltip("Minimum rotation angle in degrees")]
            public float minRotationAngle;
            [Tooltip("Maximum rotation angle in degrees")]
            public float maxRotationAngle;
        }
        [Header("Joint Configurations")]
        public JointConfig[] joints = new JointConfig[1];

        [Header("Default Settings")]
        public float defaultLinkLength = 1f;
        public float jointRadius = 0.1f;
        public float linkWidth = 0.05f;
        public Material jointMaterial;
        public Material linkMaterial;
        [Tooltip("Toggle visibility of FK visualizations (joints, links)")]
        public bool visualizeFK = true;
        [Tooltip("Toggle visibility of IK visualizations (IK target, axis cylinders)")]
        public bool visualizeIK = true;
        [Header("IK Settings")]
        public bool ikSolvering = false;

        [Range(0f, 1f)]
        public float ikWeight = 0.5f; // 0.0: 위치 중심, 1.0: 자세 중심

        public Vector3 ikTargetPosition;
        public Quaternion targetQuaternion = Quaternion.identity;
        [SerializeField]
        public Vector3 AngleChanges;
        public Material ikTargetMaterial;
        public float positionTolerance = 0.01f;
        public float orientationTolerance = 0.1f;
        public int maxIterations = 100;
        public float stepSize = 0.1f;
        public float damping = 0.01f;
        [Header("Position Offset Correction")]
        [Tooltip("Enable position offset correction based on realTcpTransform error")]
        public bool enableOffsetCorrection = false;
        [Header("Axis Cylinder Settings")]
        private float axisCylinderLength = 2.5f;
        private float axisCylinderRadius = 0.125f;
        public Material xAxisMaterial;
        public Material yAxisMaterial;
        public Material zAxisMaterial;
        public Material endSphereMaterial;
        [SerializeField]
        private bool trackOrientation = true;
        [Header("First Joint Transform")]
        public Transform firstJointTransform;
        [Tooltip("Transform of the Real TCP to compare against")]
        public Transform realTcpTransform;
        [Header("Real TCP Visualization Settings")]
        [Tooltip("Toggle visibility of Real TCP visualizations (sphere and axis cylinders)")]
        public bool visualizeRC = true;
        private GameObject realTcpSphere;
        private GameObject realTcpXAxisCylinder;
        private GameObject realTcpYAxisCylinder;
        private GameObject realTcpZAxisCylinder;
        [Header("Real TCP Visualization Materials")]
        [Tooltip("Material for Real TCP sphere")]
        public Material realTcpSphereMaterial;
        [Tooltip("Material for Real TCP X-axis cylinder")]
        public Material realTcpXAxisMaterial;
        [Tooltip("Material for Real TCP Y-axis cylinder")]
        public Material realTcpYAxisMaterial;
        [Tooltip("Material for Real TCP Z-axis cylinder")]
        public Material realTcpZAxisMaterial;
        private GameObject[] jointContainers;
        private GameObject[] startSpheres;
        private GameObject[] endSpheres;
        private GameObject[] linkCuboids;
        private GameObject ikTargetSphere;
        private GameObject xAxisCylinder;
        private GameObject yAxisCylinder;
        private GameObject zAxisCylinder;
        [Header("etc.")]
        public Vector3 ikTargetEulerAngles;
        public Quaternion targetQuat_output = Quaternion.identity;
        void Start()
        {
            InitializeRobotArm();
            if (realTcpTransform != null)
            {
                Vector3[] calculatedEndPositions;
                Quaternion[] calculatedRotations;
                CalculateEndPositions(out calculatedEndPositions, out calculatedRotations);
                Quaternion calculatedTcpRotation = calculatedRotations[calculatedRotations.Length - 1];
                Quaternion realTcpRotation = realTcpTransform.rotation;
                targetQuaternion = realTcpRotation;
                ikTargetEulerAngles = targetQuaternion.eulerAngles;
            }
        }
        void InitializeRobotArm()
        {
            if (joints == null || joints.Length == 0)
            {
                Debug.LogError("Joints array is empty or null!");
                return;
            }
            int jointCount = joints.Length;
            jointContainers = new GameObject[jointCount];
            startSpheres = new GameObject[jointCount];
            endSpheres = new GameObject[jointCount];
            linkCuboids = new GameObject[jointCount];
            Vector3 firstPos = (firstJointTransform != null) ? firstJointTransform.position :
                              (joints[0].useCustomStartPosition ? joints[0].startPosition : Vector3.zero);
            Quaternion firstRot = (firstJointTransform != null) ? firstJointTransform.rotation : Quaternion.identity;
            jointContainers[0] = CreateJointContainer(firstPos, "Joint_0", transform);
            jointContainers[0].transform.rotation = firstRot;
            startSpheres[0] = CreateSphere(firstPos, "StartSphere", jointContainers[0].transform);
            startSpheres[0].SetActive(visualizeFK);
            UpdateRobotArm();
            Vector3 lastEndPosition = endSpheres[jointCount - 1].transform.position;
            // 수정: realTcpTransform이 있다면 해당 값을 최우선으로 사용
            if (realTcpTransform != null)
            {
                ikTargetPosition = realTcpTransform.position;
                targetQuaternion = realTcpTransform.rotation;
            }
            else
            {
                // realTcpTransform이 없을 경우에만 기존 FK 결과(lastEndPosition) 사용
                ikTargetPosition = lastEndPosition;
                targetQuaternion = Quaternion.identity;
            }
            ikTargetEulerAngles = targetQuaternion.eulerAngles;
            ikTargetSphere = CreateSphere(lastEndPosition, "IKTargetSphere", transform);
            if (ikTargetMaterial != null) ikTargetSphere.GetComponent<Renderer>().material = ikTargetMaterial;
            ikTargetSphere.SetActive(visualizeIK);
            xAxisCylinder = CreateAxisCylinder("XAxisCylinder", Vector3.right, xAxisMaterial);
            yAxisCylinder = CreateAxisCylinder("YAxisCylinder", Vector3.up, yAxisMaterial);
            zAxisCylinder = CreateAxisCylinder("ZAxisCylinder", Vector3.forward, zAxisMaterial);
            InitializeRealTcpVisualization();
        }
        void Update()
        {
            if (AngleChanges != Vector3.zero)
            {
                Quaternion deltaRotation = Quaternion.Euler(AngleChanges);
                targetQuaternion *= deltaRotation;
                ikTargetEulerAngles = targetQuaternion.eulerAngles;
                AngleChanges = Vector3.zero;
            }
            targetQuat_output = targetQuaternion * Quaternion.Euler(0f, 90f, 0f);
            if (ikSolvering)
            {
                SolveJacobianIK();
            }
            CompareTcpWithReal(realTcpTransform);
            UpdateRobotArm();
            if (realTcpTransform != null)
            {
                Vector3[] calculatedEndPositions;
                Quaternion[] calculatedRotations;
                CalculateEndPositions(out calculatedEndPositions, out calculatedRotations);
                Vector3 fkEndPosition = calculatedEndPositions[calculatedEndPositions.Length - 1];
                Quaternion fkEndRotation = calculatedRotations[calculatedRotations.Length - 1];
                Vector3 ikTargetPos = ikTargetPosition;
                Quaternion ikTargetRot = targetQuat_output;
                Vector3 realTcpPos = realTcpTransform.position;
                Quaternion realTcpRot = realTcpTransform.rotation;
            }
            else
            {
                Debug.LogWarning("realTcpTransform is null, skipping debug log.");
            }
            /*
            Debug.Log($"[Robot Debug] IK Solver: {ikSolvering} | " +
                      $"IK Target Pos: {ikTargetPosition:F4} | IK Target Rot: {targetQuaternion.eulerAngles:F1} | " +
                      $"Real TCP Pos: {realTcpTransform.position:F4} | Real TCP Rot: {realTcpTransform.rotation.eulerAngles:F1} | " +
                      $"FK End Pos: {endSpheres[joints.Length - 1].transform.position:F4} | FK End Rot: {CalculateEndRotations()[joints.Length - 1].eulerAngles:F1} | " +
                      $"Pos Error: {Vector3.Distance(ikTargetPosition, endSpheres[joints.Length - 1].transform.position):F4} | " +
                      $"Rot Error: {Quaternion.Angle(targetQuaternion, CalculateEndRotations()[joints.Length - 1]):F1}°");
            */
        }
        private Quaternion[] CalculateEndRotations()
        {
            Vector3[] dummyPositions;
            Quaternion[] rotations;
            CalculateEndPositions(out dummyPositions, out rotations);
            return rotations;
        }
        public void CompareTcpWithReal(Transform realTcpTransform)
        {
            if (realTcpTransform == null)
            {
                Debug.LogError("Real TCP Transform is null in CompareTcpWithReal!");
                return;
            }
            Vector3[] calculatedEndPositions;
            Quaternion[] calculatedRotations;
            CalculateEndPositions(out calculatedEndPositions, out calculatedRotations);
            Vector3 calculatedTcpPosition = calculatedEndPositions[calculatedEndPositions.Length - 1];
            Quaternion calculatedTcpRotation = calculatedRotations[calculatedRotations.Length - 1];
            Vector3 realTcpPosition = realTcpTransform.position;
            Quaternion realTcpRotation = realTcpTransform.rotation;
            float positionDifference = Vector3.Distance(calculatedTcpPosition, realTcpPosition);
            float rotationDifference = Quaternion.Angle(calculatedTcpRotation, realTcpRotation);
        }
        public void UpdateRobotArm()
        {
            if (joints == null || joints.Length == 0) return;
            Quaternion parentRotation = (firstJointTransform != null) ? firstJointTransform.rotation : Quaternion.identity;
            Vector3 parentEndPosition = (firstJointTransform != null) ? firstJointTransform.position :
                                       (joints[0].useCustomStartPosition ? joints[0].startPosition : Vector3.zero);
            Transform parentTransform = transform;
            for (int i = 0; i < joints.Length; i++)
            {
                JointConfig config = joints[i];
                if (config.linkLength == 0) config.linkLength = defaultLinkLength;
                if (config.rotationAxis == Vector3.zero) config.rotationAxis = Vector3.up;
                if (config.orientation == Vector3.zero) config.orientation = Vector3.forward;
                config.rotationAngle = NormalizeAngle(config.rotationAngle);
                config.rotationAngle = Mathf.Clamp(config.rotationAngle,
                    config.minRotationAngle != 0 ? config.minRotationAngle : -180f,
                    config.maxRotationAngle != 0 ? config.maxRotationAngle : 180f);
                Vector3 startPos = (i == 0 && firstJointTransform != null) ? firstJointTransform.position :
                                  (config.useCustomStartPosition ? config.startPosition :
                                  (i > 0 ? parentEndPosition : Vector3.zero));
                Quaternion currentRotation;
                if (i == 0 && firstJointTransform != null)
                {
                    // 첫 번째 조인트의 경우, firstJointTransform.rotation을 무시하고 rotationAngle만 사용
                    currentRotation = config.hasDegreeOfFreedom ?
                        Quaternion.AngleAxis(config.rotationAngle, config.rotationAxis.normalized) :
                        Quaternion.identity;
                }
                else
                {
                    // 나머지 조인트는 기존 로직 유지
                    currentRotation = config.hasDegreeOfFreedom ?
                        parentRotation * Quaternion.AngleAxis(config.rotationAngle, config.rotationAxis.normalized) :
                        parentRotation;
                }
                Vector3 endPos;
                if (config.useCustomEndPosition && config.endPosition != Vector3.zero)
                {
                    endPos = config.endPosition;
                    config.linkLength = Vector3.Distance(startPos, endPos);
                }
                else
                {
                    Vector3 localLinkVector = config.orientation.normalized * config.linkLength;
                    endPos = startPos + currentRotation * localLinkVector;
                }
                if (jointContainers[i] == null)
                {
                    jointContainers[i] = CreateJointContainer(startPos, $"Joint_{i}", parentTransform);
                }
                else
                {
                    jointContainers[i].transform.position = startPos;
                    jointContainers[i].transform.rotation = currentRotation;
                    jointContainers[i].transform.localScale = Vector3.one;
                }
                if (startSpheres[i] == null)
                {
                    startSpheres[i] = CreateSphere(startPos, "StartSphere", jointContainers[i].transform);
                    startSpheres[i].SetActive(visualizeFK);
                }
                else
                {
                    startSpheres[i].transform.position = startPos;
                    startSpheres[i].transform.localScale = Vector3.one * jointRadius * 2;
                    startSpheres[i].SetActive(visualizeFK);
                }
                if (endSpheres[i] == null)
                {
                    endSpheres[i] = CreateSphere(endPos, "EndSphere", jointContainers[i].transform);
                    endSpheres[i].SetActive(visualizeFK);
                }
                else
                {
                    endSpheres[i].transform.position = endPos;
                    endSpheres[i].transform.localScale = Vector3.one * jointRadius * 2;
                    endSpheres[i].SetActive(visualizeFK);
                }
                if (linkCuboids[i] == null)
                {
                    linkCuboids[i] = CreateLink(startPos, endPos, "Box", jointContainers[i].transform);
                    linkCuboids[i].SetActive(visualizeFK);
                }
                else
                {
                    UpdateLink(linkCuboids[i], startPos, endPos);
                    linkCuboids[i].SetActive(visualizeFK);
                }
                parentRotation = currentRotation;
                parentEndPosition = endPos;
                parentTransform = jointContainers[i].transform;
                joints[i] = config;
                UpdateRealTcpVisualization();
            }
            if (ikTargetSphere != null)
            {
                ikTargetSphere.transform.position = ikTargetPosition;
                ikTargetSphere.transform.rotation = targetQuaternion;
                ikTargetSphere.SetActive(visualizeIK);
                UpdateAxisCylinder(xAxisCylinder, Vector3.right);
                UpdateAxisCylinder(yAxisCylinder, Vector3.up);
                UpdateAxisCylinder(zAxisCylinder, Vector3.forward);
            }
        }
        private void InitializeRealTcpVisualization()
        {
            realTcpSphere = CreateSphere(realTcpTransform != null ? realTcpTransform.position : Vector3.zero, "RealTcpSphere", transform);
            if (realTcpSphereMaterial != null)
            {
                realTcpSphere.GetComponent<Renderer>().material = realTcpSphereMaterial;
            }
            realTcpSphere.SetActive(visualizeRC);
            realTcpXAxisCylinder = CreateRealTcpAxisCylinder("RealTcpXAxisCylinder", Vector3.right, realTcpXAxisMaterial);
            realTcpYAxisCylinder = CreateRealTcpAxisCylinder("RealTcpYAxisCylinder", Vector3.up, realTcpYAxisMaterial);
            realTcpZAxisCylinder = CreateRealTcpAxisCylinder("RealTcpZAxisCylinder", Vector3.forward, realTcpZAxisMaterial);
        }
        private GameObject CreateRealTcpAxisCylinder(string name, Vector3 axis, Material material)
        {
            GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            cylinder.name = name;
            cylinder.transform.SetParent(realTcpSphere.transform);
            cylinder.transform.rotation = realTcpTransform != null ? realTcpTransform.rotation * Quaternion.FromToRotation(Vector3.up, axis) : Quaternion.identity;
            cylinder.transform.localScale = new Vector3(axisCylinderRadius * 2, axisCylinderLength / 2, axisCylinderRadius * 2);
            cylinder.transform.localPosition = Vector3.zero;
            if (material != null) cylinder.GetComponent<Renderer>().material = material;
            cylinder.SetActive(visualizeRC);
            string axisName = name.Contains("X") ? "X" : name.Contains("Y") ? "Y" : "Z";
            GameObject endSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            endSphere.name = $"{name}_EndSphere_{axisName}";
            endSphere.transform.SetParent(realTcpSphere.transform);
            endSphere.transform.localPosition = Quaternion.Euler(0f, -90f, 0f) * (axis * (axisCylinderLength - 1.2f));
            Vector3 inverseScale = new Vector3(
                1f / realTcpSphere.transform.lossyScale.x,
                1f / realTcpSphere.transform.lossyScale.y,
                1f / realTcpSphere.transform.lossyScale.z
            );
            endSphere.transform.localScale = inverseScale * jointRadius * 0.8f;
            if (endSphereMaterial != null) endSphere.GetComponent<Renderer>().material = endSphereMaterial;
            endSphere.SetActive(visualizeRC);
            return cylinder;
        }
        private void UpdateRealTcpVisualization()
        {
            if (realTcpSphere != null && realTcpTransform != null)
            {
                realTcpSphere.transform.position = realTcpTransform.position;
                realTcpSphere.transform.rotation = realTcpTransform.rotation;
                realTcpSphere.SetActive(visualizeRC);
                if (realTcpXAxisCylinder != null)
                {
                    realTcpXAxisCylinder.transform.rotation = realTcpTransform.rotation * Quaternion.FromToRotation(Vector3.up, Vector3.right);
                    realTcpXAxisCylinder.transform.localPosition = Vector3.zero;
                    realTcpXAxisCylinder.SetActive(visualizeRC);
                    Transform xEndSphere = realTcpSphere.transform.Find("RealTcpXAxisCylinder_EndSphere_X");
                    if (xEndSphere != null)
                    {
                        xEndSphere.localPosition = Quaternion.Euler(0f, -90f, 0f) * (Vector3.right * (axisCylinderLength - 1.2f));
                        Vector3 inverseScale = new Vector3(
                            1f / realTcpSphere.transform.lossyScale.x,
                            1f / realTcpSphere.transform.lossyScale.y,
                            1f / realTcpSphere.transform.lossyScale.z
                        );
                        xEndSphere.localScale = inverseScale * jointRadius * 0.8f;
                        if (endSphereMaterial != null) xEndSphere.GetComponent<Renderer>().material = endSphereMaterial;
                        xEndSphere.gameObject.SetActive(visualizeRC);
                    }
                }
                if (realTcpYAxisCylinder != null)
                {
                    realTcpYAxisCylinder.transform.rotation = realTcpTransform.rotation * Quaternion.FromToRotation(Vector3.up, Vector3.up);
                    realTcpYAxisCylinder.transform.localPosition = Vector3.zero;
                    realTcpYAxisCylinder.SetActive(visualizeRC);
                    Transform yEndSphere = realTcpSphere.transform.Find("RealTcpYAxisCylinder_EndSphere_Y");
                    if (yEndSphere != null)
                    {
                        yEndSphere.localPosition = Quaternion.Euler(0f, -90f, 0f) * (Vector3.up * (axisCylinderLength - 1.2f));
                        Vector3 inverseScale = new Vector3(
                            1f / realTcpSphere.transform.lossyScale.x,
                            1f / realTcpSphere.transform.lossyScale.y,
                            1f / realTcpSphere.transform.lossyScale.z
                        );
                        yEndSphere.localScale = inverseScale * jointRadius * 0.8f;
                        if (endSphereMaterial != null) yEndSphere.GetComponent<Renderer>().material = endSphereMaterial;
                        yEndSphere.gameObject.SetActive(visualizeRC);
                    }
                }
                if (realTcpZAxisCylinder != null)
                {
                    realTcpZAxisCylinder.transform.rotation = realTcpTransform.rotation * Quaternion.FromToRotation(Vector3.up, Vector3.forward);
                    realTcpZAxisCylinder.transform.localPosition = Vector3.zero;
                    realTcpZAxisCylinder.SetActive(visualizeRC);
                    Transform zEndSphere = realTcpSphere.transform.Find("RealTcpZAxisCylinder_EndSphere_Z");
                    if (zEndSphere != null)
                    {
                        zEndSphere.localPosition = Quaternion.Euler(0f, -90f, 0f) * (Vector3.forward * (axisCylinderLength - 1.2f));
                        Vector3 inverseScale = new Vector3(
                            1f / realTcpSphere.transform.lossyScale.x,
                            1f / realTcpSphere.transform.lossyScale.y,
                            1f / realTcpSphere.transform.lossyScale.z
                        );
                        zEndSphere.localScale = inverseScale * jointRadius * 0.8f;
                        if (endSphereMaterial != null) zEndSphere.GetComponent<Renderer>().material = endSphereMaterial;
                        zEndSphere.gameObject.SetActive(visualizeRC);
                    }
                }
            }
        }
        private GameObject CreateAxisCylinder(string name, Vector3 axis, Material material)
        {
            GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            cylinder.name = name;
            cylinder.transform.SetParent(ikTargetSphere.transform);
            cylinder.transform.rotation = targetQuat_output * Quaternion.FromToRotation(Vector3.up, axis);
            cylinder.transform.localScale = new Vector3(axisCylinderRadius * 2, axisCylinderLength / 2, axisCylinderRadius * 2);
            cylinder.transform.localPosition = Vector3.zero;
            if (material != null) cylinder.GetComponent<Renderer>().material = material;
            cylinder.SetActive(visualizeIK);
            string axisName = name.Contains("X") ? "X" : name.Contains("Y") ? "Y" : "Z";
            GameObject endSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            endSphere.name = $"{name}_EndSphere_{axisName}";
            endSphere.transform.SetParent(ikTargetSphere.transform);
            endSphere.transform.localPosition = axis * (axisCylinderLength - 1.2f);
            Vector3 inverseScale = new Vector3(
                1f / ikTargetSphere.transform.lossyScale.x,
                1f / ikTargetSphere.transform.lossyScale.y,
                1f / ikTargetSphere.transform.lossyScale.z
            );
            endSphere.transform.localScale = inverseScale * jointRadius * 0.8f;
            if (endSphereMaterial != null) endSphere.GetComponent<Renderer>().material = endSphereMaterial;
            endSphere.SetActive(visualizeIK);
            return cylinder;
        }
        private void UpdateAxisCylinder(GameObject cylinder, Vector3 axis)
        {
            if (cylinder == null) return;
            cylinder.transform.rotation = targetQuat_output * Quaternion.FromToRotation(Vector3.up, axis);
            cylinder.transform.localPosition = Vector3.zero;
            cylinder.SetActive(visualizeIK);
            string axisName = cylinder.name.Contains("X") ? "X" : cylinder.name.Contains("Y") ? "Y" : "Z";
            Transform endSphere = ikTargetSphere.transform.Find($"{cylinder.name}_EndSphere_{axisName}");
            if (endSphere != null)
            {
                endSphere.localPosition = axis * (axisCylinderLength - 1.2f);
                Vector3 inverseScale = new Vector3(
                    1f / ikTargetSphere.transform.lossyScale.x,
                    1f / ikTargetSphere.transform.lossyScale.y,
                    1f / ikTargetSphere.transform.lossyScale.z
                );
                endSphere.localScale = inverseScale * jointRadius * 0.8f;
                if (endSphereMaterial != null) endSphere.GetComponent<Renderer>().material = endSphereMaterial;
                endSphere.gameObject.SetActive(visualizeIK);
            }
        }
        void SolveJacobianIK()
        {
            int dofCount = 0;
            foreach (var joint in joints)
            {
                if (joint.hasDegreeOfFreedom) dofCount++;
            }
            if (dofCount == 0)
            {
                Debug.LogWarning("No degrees of freedom available for IK solving!");
                return;
            }

            int errorDim = 3 + (trackOrientation ? 3 : 0);
            for (int iter = 0; iter < maxIterations; iter++)
            {
                Vector3[] endPositions;
                Quaternion[] rotations;
                CalculateEndPositions(out endPositions, out rotations);
                Vector3 currentPos = endPositions[endPositions.Length - 1];
                Quaternion currentRot = rotations[rotations.Length - 1];

                Vector3 positionOffset = Vector3.zero;
                if (enableOffsetCorrection && realTcpTransform != null)
                {
                    Vector3 realTcpPos = realTcpTransform.position;
                    positionOffset = ikTargetPosition - realTcpPos;
                }
                Vector3 adjustedTargetPos = ikTargetPosition + positionOffset;
                Vector3 posError = adjustedTargetPos - currentPos;
                Vector3 rotError = Vector3.zero;
                float angle = 0f;

                if (trackOrientation)
                {
                    Quaternion targetRotation = targetQuaternion;
                    Quaternion rotErrorQuat = targetRotation * Quaternion.Inverse(currentRot);
                    rotErrorQuat.ToAngleAxis(out angle, out rotError);
                    if (angle > 180f) angle -= 360f;
                    rotError *= angle * Mathf.Deg2Rad;
                }

                Vector error = new Vector(errorDim);

                // --- 수정된 부분: 가중치(ikWeight) 적용 로직 ---
                // ikWeight 0.5가 기존 상태(1.0배)가 되도록 매핑
                // 0.0에 가까울수록 위치 강조(2.0 -> 0.0), 1.0에 가까울수록 자세 강조(0.0 -> 2.0)
                float posW = Mathf.Clamp01((0.5f - ikWeight) * 2.0f + 1.0f);
                float rotW = Mathf.Clamp01((ikWeight - 0.5f) * 2.0f + 1.0f);

                error[0] = posError.x * posW;
                error[1] = posError.y * posW;
                error[2] = posError.z * posW;

                if (trackOrientation)
                {
                    error[3] = rotError.x * rotW;
                    error[4] = rotError.y * rotW;
                    error[5] = rotError.z * rotW;
                }
                // ----------------------------------------------

                if (posError.sqrMagnitude < positionTolerance * positionTolerance &&
                    (!trackOrientation || Mathf.Abs(angle) < orientationTolerance))
                {
                    break;
                }

                Matrix jacobian = ComputeJacobian(endPositions, rotations, dofCount, errorDim);
                Vector deltaTheta = ComputeDampedLeastSquares(jacobian, error);

                int dofIndex = 0;
                for (int i = 0; i < joints.Length; i++)
                {
                    if (!joints[i].hasDegreeOfFreedom) continue;
                    JointConfig config = joints[i];
                    config.rotationAngle += deltaTheta[dofIndex] * stepSize;
                    config.rotationAngle = NormalizeAngle(config.rotationAngle);
                    config.rotationAngle = Mathf.Clamp(config.rotationAngle,
                        config.minRotationAngle != 0 ? config.minRotationAngle : -180f,
                        config.maxRotationAngle != 0 ? config.maxRotationAngle : 180f);
                    joints[i] = config;
                    dofIndex++;
                }
                UpdateRobotArm();
            }
        }
        Matrix ComputeJacobian(Vector3[] endPositions, Quaternion[] rotations, int dofCount, int errorDim)
        {
            Matrix jacobian = new Matrix(errorDim, dofCount);
            Vector3 endEffectorPos = endPositions[endPositions.Length - 1];
            Quaternion endEffectorRot = rotations[rotations.Length - 1];
            int dofIndex = 0;
            for (int i = 0; i < joints.Length; i++)
            {
                if (!joints[i].hasDegreeOfFreedom) continue;
                Vector3 jointPos = i == 0 && firstJointTransform != null
                    ? firstJointTransform.position
                    : (joints[i].useCustomStartPosition ? joints[i].startPosition : (i > 0 ? endPositions[i - 1] : Vector3.zero));
                Vector3 rotationAxis = joints[i].rotationAxis.normalized;
                Vector3 globalAxis = rotations[i] * rotationAxis;
                Vector3 posJacobian = Vector3.Cross(globalAxis, endEffectorPos - jointPos);
                Vector3 rotJacobian = trackOrientation ? globalAxis : Vector3.zero;
                Vector column = new Vector(errorDim);
                column[0] = posJacobian.x;
                column[1] = posJacobian.y;
                column[2] = posJacobian.z;
                if (trackOrientation)
                {
                    column[3] = rotJacobian.x;
                    column[4] = rotJacobian.y;
                    column[5] = rotJacobian.z;
                }
                jacobian.SetColumn(dofIndex, column);
                dofIndex++;
            }
            return jacobian;
        }
        Vector ComputeDampedLeastSquares(Matrix jacobian, Vector error)
        {
            Matrix jt = jacobian.Transpose();
            Matrix jtJ = jt * jacobian;
            for (int i = 0; i < jtJ.Rows; i++)
            {
                jtJ[i, i] += damping * damping;
            }
            Vector jtError = jt * error;
            return jtJ.Solve(jtError);
        }
        float NormalizeAngle(float angle)
        {
            angle = angle % 360f;
            if (angle > 180f) angle -= 360f;
            if (angle < -180f) angle += 360f;
            return angle;
        }
        GameObject CreateJointContainer(Vector3 position, string name, Transform parent)
        {
            GameObject container = new GameObject(name);
            container.transform.position = position;
            container.transform.localScale = Vector3.one;
            container.transform.SetParent(parent);
            return container;
        }
        GameObject CreateSphere(Vector3 position, string name, Transform parent)
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = name;
            sphere.transform.position = position;
            sphere.transform.localScale = Vector3.one * jointRadius * 2;
            sphere.transform.SetParent(parent);
            if (jointMaterial != null && (name != "IKTargetSphere" && name != "EndEffectorSphere"))
                sphere.GetComponent<Renderer>().material = jointMaterial;
            return sphere;
        }
        GameObject CreateLink(Vector3 start, Vector3 end, string name, Transform parent)
        {
            GameObject link = GameObject.CreatePrimitive(PrimitiveType.Cube);
            link.name = name;
            link.transform.SetParent(parent);
            UpdateLink(link, start, end);
            if (linkMaterial != null) link.GetComponent<Renderer>().material = linkMaterial;
            return link;
        }
        void UpdateLink(GameObject link, Vector3 start, Vector3 end)
        {
            Vector3 direction = (end - start).normalized;
            float length = Vector3.Distance(start, end);
            link.transform.position = start + direction * (length / 2f);
            link.transform.rotation = Quaternion.FromToRotation(Vector3.forward, direction);
            link.transform.localScale = new Vector3(linkWidth, linkWidth, length);
        }
        public void CalculateEndPositions(out Vector3[] calculatedEndPositions, out Quaternion[] calculatedRotations)
        {
            calculatedEndPositions = new Vector3[joints.Length];
            calculatedRotations = new Quaternion[joints.Length];
            Quaternion parentRotation = (firstJointTransform != null) ? firstJointTransform.rotation : Quaternion.identity;
            Vector3 parentEndPosition = (firstJointTransform != null) ? firstJointTransform.position :
                                       (joints[0].useCustomStartPosition ? joints[0].startPosition : Vector3.zero);
            for (int i = 0; i < joints.Length; i++)
            {
                JointConfig config = joints[i];
                if (config.linkLength == 0) config.linkLength = defaultLinkLength;
                if (config.rotationAxis == Vector3.zero) config.rotationAxis = Vector3.up;
                if (config.orientation == Vector3.zero) config.orientation = Vector3.forward;
                config.rotationAngle = NormalizeAngle(config.rotationAngle);
                config.rotationAngle = Mathf.Clamp(config.rotationAngle,
                    config.minRotationAngle != 0 ? config.minRotationAngle : -180f,
                    config.maxRotationAngle != 0 ? config.maxRotationAngle : 180f);
                Vector3 startPos = (i == 0 && firstJointTransform != null) ? firstJointTransform.position :
                                  (config.useCustomStartPosition ? config.startPosition :
                                  (i > 0 ? parentEndPosition : Vector3.zero));
                Quaternion currentRotation = config.hasDegreeOfFreedom ?
                    parentRotation * Quaternion.AngleAxis(config.rotationAngle, config.rotationAxis.normalized) :
                    parentRotation;
                Vector3 endPos;
                if (config.useCustomEndPosition && config.endPosition != Vector3.zero)
                {
                    endPos = config.endPosition;
                    config.linkLength = Vector3.Distance(startPos, endPos);
                }
                else
                {
                    Vector3 localLinkVector = config.orientation.normalized * config.linkLength;
                    endPos = startPos + currentRotation * localLinkVector;
                }
                calculatedEndPositions[i] = endPos;
                calculatedRotations[i] = currentRotation;
                parentRotation = currentRotation;
                parentEndPosition = endPos;
                joints[i] = config;
            }
        }
    }
    public class Matrix
    {
        private float[,] data;
        public int Rows { get; private set; }
        public int Cols { get; private set; }
        public Matrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
            data = new float[rows, cols];
        }
        public float this[int row, int col]
        {
            get => data[row, col];
            set => data[row, col] = value;
        }
        public void SetColumn(int col, Vector vec)
        {
            for (int i = 0; i < Rows && i < vec.Size; i++)
            {
                data[i, col] = vec[i];
            }
        }
        public Matrix Transpose()
        {
            Matrix result = new Matrix(Cols, Rows);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[j, i] = data[i, j];
            return result;
        }
        public static Matrix operator *(Matrix a, Matrix b)
        {
            if (a.Cols != b.Rows)
            {
                Debug.LogError($"Matrix multiplication invalid: {a.Rows}x{a.Cols} cannot multiply with {b.Rows}x{b.Cols}");
                return new Matrix(a.Rows, b.Cols);
            }
            Matrix result = new Matrix(a.Rows, b.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < b.Cols; j++)
                    for (int k = 0; k < a.Cols; k++)
                        result[i, j] += a[i, k] * b[k, j];
            return result;
        }
        public static Vector operator *(Matrix m, Vector v)
        {
            if (m.Cols != v.Size)
            {
                Debug.LogError($"Matrix-Vector multiplication invalid: {m.Rows}x{m.Cols} cannot multiply with vector of size {v.Size}");
                return new Vector(m.Rows);
            }
            Vector result = new Vector(m.Rows);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Cols; j++)
                    result[i] += m[i, j] * v[j];
            return result;
        }
        public Vector Solve(Vector b)
        {
            if (Rows != b.Size)
            {
                Debug.LogError($"Solve invalid: Matrix rows ({Rows}) must match vector size ({b.Size})");
                return new Vector(Rows);
            }
            int n = Rows;
            Matrix augmented = new Matrix(n, n + 1);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    augmented[i, j] = data[i, j];
                augmented[i, n] = b[i];
            }
            for (int i = 0; i < n; i++)
            {
                float pivot = augmented[i, i];
                if (Mathf.Abs(pivot) < 0.0001f)
                {
                    Debug.LogWarning("Singular matrix detected, returning zero vector");
                    return new Vector(n);
                }
                for (int j = i; j <= n; j++)
                    augmented[i, j] /= pivot;
                for (int k = 0; k < n; k++)
                {
                    if (k != i)
                    {
                        float factor = augmented[k, i];
                        for (int j = i; j <= n; j++)
                            augmented[k, j] -= factor * augmented[i, j];
                    }
                }
            }
            Vector result = new Vector(n);
            for (int i = 0; i < n; i++)
                result[i] = augmented[i, n];
            return result;
        }
    }
    public class Vector
    {
        private float[] data;
        public int Size { get; private set; }
        public Vector(int size)
        {
            Size = size;
            data = new float[size];
        }
        public float this[int index]
        {
            get
            {
                if (index < 0 || index >= Size)
                {
                    Debug.LogError($"Vector index out of range: {index}");
                    return 0f;
                }
                return data[index];
            }
            set
            {
                if (index < 0 || index >= Size)
                {
                    Debug.LogError($"Vector index out of range: {index}");
                    return;
                }
                data[index] = value;
            }
        }
    }
}