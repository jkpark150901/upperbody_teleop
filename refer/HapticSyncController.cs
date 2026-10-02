using UnityEngine;
using System.IO;
using RobotArm;
using Mujoco;
using System.Collections;

namespace RobotArm
{
    public class HapticSyncController : MonoBehaviour
    {
        [Header("Haptic Hand References")]
        [Tooltip("Haptic hand's Wrist Transform for position and rotation")]
        public Transform wristTransform;
        [Tooltip("Haptic hand's ThumbDistal Transform")]
        public Transform thumbDistalTransform;
        [Tooltip("Haptic hand's IndexDistal Transform")]
        public Transform indexDistalTransform;

        [Header("Robot Arm Reference")]
        [Tooltip("Reference to RobotArmFK component")]
        public RobotArmFK robotArmFK;

        [Header("Gripper Actuator References")]
        [Tooltip("True applies original values, False applies inverted values (* -1)")]
        public bool leftornot = true;
        [Tooltip("List of 6 MjActuator components for Gripper")]
        public Mujoco.MjActuator[] gripperActuators = new Mujoco.MjActuator[6];

        [Header("Initial Target Transform")]
        [Tooltip("Optional Transform to set initial IK target position and rotation")]
        public Transform initialTargetTransform;

        [Header("Root Transform")]
        [Tooltip("Root Transform of the robot arm to limit ikTargetPosition within MaxRadius")]
        public Transform rootTransform;

        [Header("Gripper Control Settings")]
        [Tooltip("Minimum Euclidean distance between ThumbDistal and IndexDistal (in meters)")]
        public float minDistance = 0.017f;
        [Tooltip("Maximum Euclidean distance between ThumbDistal and IndexDistal (in meters)")]
        public float maxDistance = 0.1f;

        // 개별 액츄에이터의 폐쇄(Closed) 목표값 설정 (요구사항 반영)
        private float[] gripperClosedValues = new float[] { -10f, -17f, 15f, 17f, 15f, 17f };
        private float gripperOpenValue = 0f;

        [Header("Tracking Toggles")]
        [Tooltip("Enable tracking of Wrist position to ikTargetPosition")]
        public bool trackPosition = true;
        [Tooltip("Enable tracking of Wrist rotation to targetQuaternion")]
        public bool trackRotation = true;
        [Tooltip("Enable tracking of Gripper control based on finger distance")]
        public bool trackGripper = true;

        [Header("Position Tracking Settings")]
        [Tooltip("Scale factor for movement (applied to all axes)")]
        public float scale = 1.0f;

        [Header("Smoothing Settings")]
        [Tooltip("Smoothing factor for low-pass filter (0 to 1, lower is smoother)")]
        [Range(0f, 1f)]
        public float smoothingFactor = 0.1f;

        [Header("Logging Settings")]
        [Tooltip("Enable logging of data to CSV file")]
        public bool toCSV = false;

        public float MaxRadius = 0.4f;
        private Vector3 initialWristPosition;
        private Quaternion initialWristRotation;
        public Vector3 initialIkTargetPosition;
        public Quaternion initialIkTargetRotation;
        private bool isCalibrated = false;
        private int spacePressCount = 0;
        private bool FirstTimeExe = false;

        // 필터링된 값 저장 변수
        private Vector3 filteredWristDelta = Vector3.zero;
        private Quaternion filteredWristDeltaRotation = Quaternion.identity;
        private float filteredDistance = 0f;

        // CSV 로깅 관련 변수
        private string filePath;
        private StreamWriter writer;
        private bool isRecording;

        private void Start()
        {
            // 초기 IK 타겟 위치 및 회전 설정
            if (initialTargetTransform != null)
            {
                initialIkTargetPosition = initialTargetTransform.position;
                initialIkTargetRotation = initialTargetTransform.rotation;
            }
            else
            {
                initialIkTargetPosition = new Vector3(-1.35f, 1.5f, -1.5f);
                initialIkTargetRotation = Quaternion.Euler(0f, 0f, 0f);
            }

            // 초기 검증
            if (wristTransform == null)
            {
                Debug.LogError("Wrist Transform is not assigned in HapticSyncController!");
                enabled = false;
                return;
            }
            if (thumbDistalTransform == null || indexDistalTransform == null)
            {
                Debug.LogError("ThumbDistal or IndexDistal Transform is not assigned in HapticSyncController!");
                enabled = false;
                return;
            }
            if (robotArmFK == null)
            {
                Debug.LogError("RobotArmFK component is not assigned in HapticSyncController!");
                enabled = false;
                return;
            }
            // 배열 크기 및 할당 체크
            if (gripperActuators == null || gripperActuators.Length < 6)
            {
                Debug.LogError("Please assign 6 Gripper Actuators in the inspector!");
                enabled = false;
                return;
            }
            if (rootTransform == null)
            {
                Debug.LogError("Root Transform is not assigned in HapticSyncController!");
                enabled = false;
                return;
            }

            // 초기 필터링 값 설정
            filteredDistance = minDistance;

            // CSV 로깅 초기화
            if (toCSV)
            {
                StartRecording();
            }
        }

        private void Update()
        {
            if (FirstTimeExe == false)
            {
                robotArmFK.ikTargetPosition = initialIkTargetPosition;
                robotArmFK.targetQuaternion = initialIkTargetRotation;
                FirstTimeExe = true;
            }

            // 스페이스바 입력 처리
            if (Input.GetKeyDown(KeyCode.Space))
            {
                StartCoroutine(DelayedCalibration(3f));
            }

            // 위치 추적 (캘리브레이션 완료 후에만)
            if (trackPosition && isCalibrated)
            {
                // wristTransform의 초기 위치로부터의 상대적 이동 계산
                Vector3 wristDelta = wristTransform.position - initialWristPosition;

                // 로우 패스 필터 적용
                filteredWristDelta = Vector3.Lerp(filteredWristDelta, wristDelta, smoothingFactor);

                // 단일 배율 적용
                Vector3 scaledDelta = filteredWristDelta * scale;

                // 초기 ikTargetPosition에 배율 적용된 이동 추가
                Vector3 targetPosition = initialIkTargetPosition + scaledDelta;

                // rootTransform을 기준으로 MaxRadius 내로 제한 (드리프트 방지 로직)
                Vector3 vectorFromRoot = targetPosition - rootTransform.position;
                if (vectorFromRoot.magnitude > MaxRadius)
                {
                    vectorFromRoot = vectorFromRoot.normalized * MaxRadius;
                }
                robotArmFK.ikTargetPosition = rootTransform.position + vectorFromRoot;
            }

            // 회전 추적 (캘리브레이션 완료 후에만)
            if (trackRotation && isCalibrated)
            {
                // wristTransform의 초기 회전으로부터의 상대적 회전 계산 (쿼터니언)
                Quaternion wristDeltaRotation = wristTransform.rotation * Quaternion.Inverse(initialWristRotation);

                // 오일러 각도로 변환하여 각 축 반전 (-롤, -피치, -요)
                Vector3 eulerRotation = wristDeltaRotation.eulerAngles;
                Vector3 inverseEulerRotation = new Vector3(
                    NormalizeAngle(eulerRotation.x),
                    NormalizeAngle(eulerRotation.y),
                    NormalizeAngle(eulerRotation.z)
                );

                // 반전된 오일러 각도를 쿼터니언으로 변환
                Quaternion inverseWristDeltaRotation = Quaternion.Euler(inverseEulerRotation);

                // 로우 패스 필터 적용 (쿼터니언)
                filteredWristDeltaRotation = Quaternion.Slerp(filteredWristDeltaRotation, inverseWristDeltaRotation, smoothingFactor);

                // 초기 targetQuaternion에 반전된 상대적 회전 적용
                robotArmFK.targetQuaternion = initialIkTargetRotation * filteredWristDeltaRotation;
            }

            // 그리퍼 추적
            if (trackGripper)
            {
                float distance = Vector3.Distance(thumbDistalTransform.position, indexDistalTransform.position);
                // 로우 패스 필터 적용
                filteredDistance = Mathf.Lerp(filteredDistance, distance, smoothingFactor);

                // 손가락 간의 거리를 0(최소) ~ 1(최대)로 정규화 (minDistance일 때 닫힘, maxDistance일 때 열림)
                float t = Mathf.InverseLerp(minDistance, maxDistance, filteredDistance);

                // 반전 계수 결정 (leftornot이 true면 1, false면 -1)
                float inversionMultiplier = leftornot ? 1f : -1f;

                for (int i = 0; i < gripperActuators.Length; i++)
                {
                    if (gripperActuators[i] == null) continue;

                    // i가 6개를 넘어가지 않도록 방어 코드 (0~5 인덱스 사용)
                    if (i < gripperClosedValues.Length)
                    {
                        // 반전 계수가 적용된 목표 폐쇄 값
                        float targetClosedValue = gripperClosedValues[i] * inversionMultiplier;

                        // t=0 (minDistance)일 때 targetClosedValue, t=1 (maxDistance)일 때 OpenValue(0)
                        float controlValue = Mathf.Lerp(targetClosedValue, gripperOpenValue, t);
                        gripperActuators[i].Control = controlValue;
                    }
                }
            }

            // CSV 로깅
            if (toCSV && isRecording)
            {
                WriteData();
                writer.Flush();
            }
        }

        private IEnumerator DelayedCalibration(float delay)
        {
            yield return new WaitForSeconds(delay);

            spacePressCount++;
            initialWristPosition = wristTransform.position;
            initialWristRotation = wristTransform.rotation;

            // 캘리브레이션 및 재캘리브레이션 시 필터링 값 즉시 강제 동기화 (드리프트 방지)
            filteredWristDelta = Vector3.zero;
            filteredWristDeltaRotation = Quaternion.identity;

            if (spacePressCount == 1)
            {
                // 첫 번째 스페이스바 입력: 초기 IK 설정 및 캘리브레이션 활성화
                robotArmFK.ikTargetPosition = initialIkTargetPosition;
                robotArmFK.targetQuaternion = initialIkTargetRotation;
                isCalibrated = true;
                Debug.Log("Calibration completed. IK control enabled.");
            }
            else
            {
                // 두 번째 이후 스페이스바 입력: 캘리브레이션만 갱신
                Debug.Log("Recalibration: initialWristPosition and initialWristRotation updated.");
            }
        }

        private void OnApplicationQuit()
        {
            // 애플리케이션 종료 시 CSV 파일 닫기
            StopRecording();
        }

        private void StartRecording()
        {
            // LeftHand 디렉토리 생성
            string logDir = Path.Combine(Application.dataPath, "LeftHand");
            if (!Directory.Exists(logDir))
            {
                Directory.CreateDirectory(logDir);
            }

            // CSV 파일 이름 생성 및 초기화
            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            filePath = Path.Combine(logDir, $"HandData_{timestamp}.csv");
            writer = new StreamWriter(filePath, false);
            WriteHeader();
            isRecording = true;
            Debug.Log($"CSV 로깅 시작: {filePath}");
        }

        private void StopRecording()
        {
            if (writer != null)
            {
                writer.Close();
                writer = null;
            }
            isRecording = false;
            Debug.Log("CSV 로깅 중단");
        }

        private void WriteHeader()
        {
            // CSV 헤더 작성
            string[] headers = new string[]
            {
                "Time",
                "Wrist.position.x", "Wrist.position.y", "Wrist.position.z",
                "robotArmFK.ikTargetPosition.x", "robotArmFK.ikTargetPosition.y", "robotArmFK.ikTargetPosition.z",
                "Wrist.rotation.w", "Wrist.rotation.x", "Wrist.rotation.y", "Wrist.rotation.z",
                "robotArmFK.targetQuaternion.w", "robotArmFK.targetQuaternion.x", "robotArmFK.targetQuaternion.y", "robotArmFK.targetQuaternion.z",
                "FingerDistance", "gripperActuator[0].Control" // 첫번째 액츄에이터 기준 기록
            };
            writer.WriteLine(string.Join(",", headers));
        }

        private void WriteData()
        {
            // CSV 데이터 작성
            string[] data = new string[]
            {
                System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                wristTransform.position.x.ToString(),
                wristTransform.position.y.ToString(),
                wristTransform.position.z.ToString(),
                robotArmFK.ikTargetPosition.x.ToString(),
                robotArmFK.ikTargetPosition.y.ToString(),
                robotArmFK.ikTargetPosition.z.ToString(),
                wristTransform.rotation.w.ToString(),
                wristTransform.rotation.x.ToString(),
                wristTransform.rotation.y.ToString(),
                wristTransform.rotation.z.ToString(),
                robotArmFK.targetQuaternion.w.ToString(),
                robotArmFK.targetQuaternion.x.ToString(),
                robotArmFK.targetQuaternion.y.ToString(),
                robotArmFK.targetQuaternion.z.ToString(),
                Vector3.Distance(thumbDistalTransform.position, indexDistalTransform.position).ToString(),
                (gripperActuators.Length > 0 && gripperActuators[0] != null) ? gripperActuators[0].Control.ToString() : "0"
            };
            writer.WriteLine(string.Join(",", data));
        }

        // 오일러 각도를 -180~180도로 정규화하는 헬퍼 메서드
        private float NormalizeAngle(float angle)
        {
            angle = angle % 360f;
            if (angle > 180f) angle -= 360f;
            if (angle < -180f) angle += 360f;
            return angle;
        }
    }
}