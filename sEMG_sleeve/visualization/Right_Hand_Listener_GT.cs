using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Globalization;

public class Right_Hand_Listener_GT : MonoBehaviour
{
    [Header("Network Settings")]
    public int port = 5017;  // Ground-truth hand stream arrives on 5017

    [Header("Hierarchy Mapping (26 Bones)")]
    [Tooltip("Order: Wrist, Palm, Index(5), Middle(5), Ring(5), Little(5), Thumb(4)")]
    public Transform[] boneHierarchy;

    [Header("Debug")]
    public bool showDebugInfo = true;
    public int debugJointIndex = 3; // Index Proximal for debugging

    private UdpClient client;
    private Thread receiveThread;
    private float[] latestData;
    private float[] lastValidRotations;
    private bool hasNewData = false;
    private object dataLock = new object();
    private int frameCount = 0;

    void Start()
    {
        // Initialize the array to match the expected 78 floats (26 joints * 3 axes)
        latestData = new float[78];
        lastValidRotations = new float[78];

        for (int i = 0; i < latestData.Length; i++)
        {
            latestData[i] = float.NaN;
        }

        // Validate bone hierarchy
        if (boneHierarchy == null || boneHierarchy.Length != 26)
        {
            Debug.LogError("Bone hierarchy must contain exactly 26 transforms!");
            enabled = false;
            return;
        }

        // Initialize lastValidRotations with current Unity rotations
        for (int i = 0; i < boneHierarchy.Length; i++)
        {
            if (boneHierarchy[i] != null)
            {
                Vector3 currentEuler = boneHierarchy[i].localEulerAngles;
                int startIndex = i * 3;
                lastValidRotations[startIndex] = currentEuler.x;
                lastValidRotations[startIndex + 1] = currentEuler.y;
                lastValidRotations[startIndex + 2] = currentEuler.z;
            }
        }

        // Start background listener thread
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();

        Debug.Log($"Right Hand Ground Truth Listener started on port {port}");
    }

    private void ReceiveData()
    {
        try
        {
            client = new UdpClient(port);
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);

            Debug.Log("UDP Client initialized successfully");

            while (true)
            {
                try
                {
                    byte[] data = client.Receive(ref anyIP);
                    string text = Encoding.UTF8.GetString(data);
                    string[] items = text.Split(',');

                    // Thread-safe data update
                    lock (dataLock)
                    {
                        // Parse strings to floats, handling "nan" explicitly
                        for (int i = 0; i < items.Length && i < latestData.Length; i++)
                        {
                            string val = items[i].Trim().ToLower();
                            if (val == "nan")
                            {
                                latestData[i] = float.NaN;
                            }
                            else
                            {
                                // Use InvariantCulture to handle decimal points correctly
                                if (float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out float result))
                                {
                                    latestData[i] = result;
                                }
                                else
                                {
                                    latestData[i] = float.NaN;
                                }
                            }
                        }
                        hasNewData = true;
                    }
                }
                catch (SocketException se)
                {
                    Debug.LogWarning("Socket Exception: " + se.Message);
                }
                catch (Exception e)
                {
                    Debug.LogWarning("UDP Receive Error: " + e.Message);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("UDP Client initialization failed: " + e.Message);
        }
    }

    void Update()
    {
        if (!hasNewData || boneHierarchy == null) return;

        lock (dataLock)
        {
            frameCount++;

            for (int i = 0; i < boneHierarchy.Length; i++)
            {
                if (boneHierarchy[i] == null) continue;

                int startIndex = i * 3;
                if (startIndex + 2 < latestData.Length)
                {
                    // Check each axis for NaN. If NaN, use last valid value.
                    float x = float.IsNaN(latestData[startIndex]) ? lastValidRotations[startIndex] : latestData[startIndex];
                    float y = float.IsNaN(latestData[startIndex + 1]) ? lastValidRotations[startIndex + 1] : latestData[startIndex + 1];
                    float z = float.IsNaN(latestData[startIndex + 2]) ? lastValidRotations[startIndex + 2] : latestData[startIndex + 2];

                    // Clamp X rotation between 0 and 90 ONLY for finger joints (index 2 and above)
                    if (i >= 2)
                    {
                        x = Mathf.Clamp(x, 0f, 90f);
                    }

                    // Store the values we're actually using
                    lastValidRotations[startIndex] = x;
                    lastValidRotations[startIndex + 1] = y;
                    lastValidRotations[startIndex + 2] = z;

                    boneHierarchy[i].localRotation = Quaternion.Euler(x, y, z);

                    // Debug output for specified joint
                    if (showDebugInfo && i == debugJointIndex && frameCount % 30 == 0)
                    {
                        Debug.Log($"Joint {i} ({boneHierarchy[i].name}): X={x:F2}° Y={y:F2}° Z={z:F2}°");
                    }
                }
            }

            hasNewData = false; // Reset flag after processing
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
        }
        if (client != null)
        {
            client.Close();
        }
        Debug.Log("Right Hand Ground Truth Listener stopped");
    }

    void OnDestroy()
    {
        OnApplicationQuit();
    }
}