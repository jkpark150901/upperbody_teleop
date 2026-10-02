using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Mujoco;

public class FingerAngleLogger : MonoBehaviour
{
    // 이동 방식 선택을 위한 열거형
    public enum MoveMode { Random, Sequential, RandomDiscrete }
    [SerializeField, Tooltip("이동 방식: Random, Sequential 또는 RandomDiscrete")]
    private MoveMode moveMode = MoveMode.Random;

    [Header("Hand Reference")]
    public Transform Wrist;

    [Header("Index Finger")]
    public Transform IndexMetacarpal;
    public Transform IndexProximal;
    public Transform IndexIntermediate;
    public Transform IndexDistal;

    [Header("Middle Finger")]
    public Transform MiddleMetacarpal;
    public Transform MiddleProximal;
    public Transform MiddleIntermediate;
    public Transform MiddleDistal;

    [Header("Ring Finger")]
    public Transform RingMetacarpal;
    public Transform RingProximal;
    public Transform RingIntermediate;
    public Transform RingDistal;

    [Header("Little Finger")]
    public Transform LittleMetacarpal;
    public Transform LittleProximal;
    public Transform LittleIntermediate;
    public Transform LittleDistal;

    [Header("Thumb")]
    public Transform ThumbMetacarpal;
    public Transform ThumbProximal;
    public Transform ThumbDistal;

    [Header("MuJoCo Actuators")]
    [SerializeField, Tooltip("Base Actuator")]
    private MjActuator baseActuator;

    [SerializeField, Tooltip("Shoulder Actuator")]
    private MjActuator shoulderActuator;

    [SerializeField, Tooltip("Elbow Actuator")]
    private MjActuator elbowActuator;

    [SerializeField, Tooltip("Wrist1 Actuator")]
    private MjActuator wrist1Actuator;

    [SerializeField, Tooltip("Wrist2 Actuator")]
    private MjActuator wrist2Actuator;

    [SerializeField, Tooltip("Wrist3 Actuator")]
    private MjActuator wrist3Actuator;

    [SerializeField, Tooltip("Finger Actuator")]
    private MjActuator fingerActuator;

    private string filePath;
    private StreamWriter writer;
    private bool isRecording;

    void Start()
    {
        // LeftHand 디렉토리 생성
        string logDir = Path.Combine(Application.dataPath, "LeftHand");
        if (!Directory.Exists(logDir))
        {
            Directory.CreateDirectory(logDir);
        }

        // 액추에이터 할당 확인
        if (baseActuator == null)
        {
            Debug.LogWarning("Base Actuator가 할당되지 않았습니다.");
        }
        if (shoulderActuator == null)
        {
            Debug.LogWarning("Shoulder Actuator가 할당되지 않았습니다.");
        }
        if (elbowActuator == null)
        {
            Debug.LogWarning("Elbow Actuator가 할당되지 않았습니다.");
        }
        if (wrist1Actuator == null)
        {
            Debug.LogWarning("Wrist1 Actuator가 할당되지 않았습니다.");
        }
        if (wrist2Actuator == null)
        {
            Debug.LogWarning("Wrist2 Actuator가 할당되지 않았습니다.");
        }
        if (wrist3Actuator == null)
        {
            Debug.LogWarning("Wrist3 Actuator가 할당되지 않았습니다.");
        }
        if (fingerActuator == null)
        {
            Debug.LogWarning("Finger Actuator가 할당되지 않았습니다.");
        }

        // RandomDiscrete 모드일 경우 CSV 파일 생성 및 초기화
        if (moveMode == MoveMode.RandomDiscrete)
        {
            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            filePath = Path.Combine(Application.dataPath, "LeftHand", $"HandData_{timestamp}.csv");
            writer = new StreamWriter(filePath, false);
            WriteHeader();
            writer.Flush();
            isRecording = true;
            Debug.Log($"RandomDiscrete 모드: 데이터 기록 시작 - {filePath}");
        }
    }

    void Update()
    {
        if (moveMode == MoveMode.Random || moveMode == MoveMode.Sequential)
        {
            // Random 또는 Sequential 모드: 기존 동작
            if (Input.GetKeyDown(KeyCode.Space) && !isRecording)
            {
                // 스페이스 바를 누르면 파일 생성 및 기록 시작
                StartRecording();
            }
            else if (Input.GetKeyUp(KeyCode.Space) && isRecording)
            {
                // 스페이스 바를 떼면 기록 중단 및 파일 닫기
                StopRecording();
            }

            // 기록 중일 때만 데이터 기록
            if (isRecording)
            {
                WriteJointData();
                writer.Flush();
            }
        }
        else if (moveMode == MoveMode.RandomDiscrete)
        {
            // RandomDiscrete 모드: 스페이스 바 누를 때만 데이터 기록
            if (Input.GetKeyDown(KeyCode.Space))
            {
                WriteJointData();
                writer.Flush();
                Debug.Log("RandomDiscrete 모드: 데이터 행 추가");
            }
        }
    }

    void OnApplicationQuit()
    {
        // 애플리케이션 종료 시 파일 닫기
        StopRecording();
    }

    private void StartRecording()
    {
        // 현재 날짜와 시간으로 CSV 파일 이름 생성
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        filePath = Path.Combine(Application.dataPath, "LeftHand", $"HandData_{timestamp}.csv");

        // CSV 파일 초기화 및 헤더 작성
        writer = new StreamWriter(filePath, false);
        WriteHeader();
        writer.Flush();
        isRecording = true;
        Debug.Log($"데이터 기록 시작: {filePath}");
    }

    private void StopRecording()
    {
        if (writer != null)
        {
            writer.Close();
            writer = null;
        }
        isRecording = false;
        Debug.Log("데이터 기록 중단");
    }

    private void WriteHeader()
    {
        // CSV 헤더 구성
        List<string> headers = new List<string>();

        // 손목 위치 및 쿼터니언
        headers.AddRange(new string[] {
            "Wrist_PosX", "Wrist_PosY", "Wrist_PosZ",
            "Wrist_RotX", "Wrist_RotY", "Wrist_RotZ", "Wrist_RotW"
        });

        // 검지 손가락 관절 (쿼터니언만)
        AddRotationHeaders(headers, "IndexMetacarpal");
        AddRotationHeaders(headers, "IndexProximal");
        AddRotationHeaders(headers, "IndexIntermediate");
        AddRotationHeaders(headers, "IndexDistal");

        // 중지 손가락 관절 (쿼터니언만)
        AddRotationHeaders(headers, "MiddleMetacarpal");
        AddRotationHeaders(headers, "MiddleProximal");
        AddRotationHeaders(headers, "MiddleIntermediate");
        AddRotationHeaders(headers, "MiddleDistal");

        // 약지 손가락 관절 (쿼터니언만)
        AddRotationHeaders(headers, "RingMetacarpal");
        AddRotationHeaders(headers, "RingProximal");
        AddRotationHeaders(headers, "RingIntermediate");
        AddRotationHeaders(headers, "RingDistal");

        // 소지 손가락 관절 (쿼터니언만)
        AddRotationHeaders(headers, "LittleMetacarpal");
        AddRotationHeaders(headers, "LittleProximal");
        AddRotationHeaders(headers, "LittleIntermediate");
        AddRotationHeaders(headers, "LittleDistal");

        // 엄지 손가락 관절 (쿼터니언만)
        AddRotationHeaders(headers, "ThumbMetacarpal");
        AddRotationHeaders(headers, "ThumbProximal");
        AddRotationHeaders(headers, "ThumbDistal");

        // 출력값: 액추에이터 제어값
        headers.AddRange(new string[] {
            "BaseActuator_Control",
            "ShoulderActuator_Control",
            "ElbowActuator_Control",
            "Wrist1Actuator_Control",
            "Wrist2Actuator_Control",
            "Wrist3Actuator_Control",
            "FingerActuator_Control"
        });

        writer.WriteLine(string.Join(",", headers));
    }

    private void AddRotationHeaders(List<string> headers, string jointName)
    {
        headers.AddRange(new string[] {
            $"{jointName}_RotX", $"{jointName}_RotY", $"{jointName}_RotZ", $"{jointName}_RotW"
        });
    }

    private void WriteJointData()
    {
        // 관절 데이터 기록
        List<string> data = new List<string>();

        // 손목 데이터 (위치 + 쿼터니언)
        AddJointData(data, Wrist, true);

        // 검지 손가락 데이터 (쿼터니언만)
        AddJointData(data, IndexMetacarpal, false);
        AddJointData(data, IndexProximal, false);
        AddJointData(data, IndexIntermediate, false);
        AddJointData(data, IndexDistal, false);

        // 중지 손가락 데이터 (쿼터니언만)
        AddJointData(data, MiddleMetacarpal, false);
        AddJointData(data, MiddleProximal, false);
        AddJointData(data, MiddleIntermediate, false);
        AddJointData(data, MiddleDistal, false);

        // 약지 손가락 데이터 (쿼터니언만)
        AddJointData(data, RingMetacarpal, false);
        AddJointData(data, RingProximal, false);
        AddJointData(data, RingIntermediate, false);
        AddJointData(data, RingDistal, false);

        // 소지 손가락 데이터 (쿼터니언만)
        AddJointData(data, LittleMetacarpal, false);
        AddJointData(data, LittleProximal, false);
        AddJointData(data, LittleIntermediate, false);
        AddJointData(data, LittleDistal, false);

        // 엄지 손가락 데이터 (쿼터니언만)
        AddJointData(data, ThumbMetacarpal, false);
        AddJointData(data, ThumbProximal, false);
        AddJointData(data, ThumbDistal, false);

        // 출력값: 액추에이터 제어값
        data.Add(baseActuator != null ? baseActuator.Control.ToString() : "0");
        data.Add(shoulderActuator != null ? shoulderActuator.Control.ToString() : "0");
        data.Add(elbowActuator != null ? elbowActuator.Control.ToString() : "0");
        data.Add(wrist1Actuator != null ? wrist1Actuator.Control.ToString() : "0");
        data.Add(wrist2Actuator != null ? wrist2Actuator.Control.ToString() : "0");
        data.Add(wrist3Actuator != null ? wrist3Actuator.Control.ToString() : "0");
        data.Add(fingerActuator != null ? fingerActuator.Control.ToString() : "0");

        writer.WriteLine(string.Join(",", data));
    }

    private void AddJointData(List<string> data, Transform joint, bool includePosition)
    {
        if (joint != null)
        {
            if (includePosition)
            {
                // 위치 (x, y, z)
                data.Add(joint.position.x.ToString());
                data.Add(joint.position.y.ToString());
                data.Add(joint.position.z.ToString());
            }
            // 쿼터니언 회전 (x, y, z, w)
            data.Add(joint.rotation.x.ToString());
            data.Add(joint.rotation.y.ToString());
            data.Add(joint.rotation.z.ToString());
            data.Add(joint.rotation.w.ToString());
        }
        else
        {
            // 관절이 null인 경우 기본값(0) 추가
            if (includePosition)
            {
                data.AddRange(new string[] { "0", "0", "0" });
            }
            data.AddRange(new string[] { "0", "0", "0", "0" });
        }
    }
}